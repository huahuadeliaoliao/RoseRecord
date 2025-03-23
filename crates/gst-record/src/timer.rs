// timer.rs
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use log::{debug, error, warn};

use crate::{RecorderError, RecorderState, Result};

/// Timer state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimerState {
    Ready,     // Timer created but not yet started
    Running,   // Timer is running
    Paused,    // Timer is paused
    Completed, // Timer has completed its duration
    Cancelled, // Timer was cancelled by user
}

/// Timer event types for history
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimerEventType {
    Created,   // Timer was created
    Started,   // Timer was started
    Paused,    // Timer was paused
    Resumed,   // Timer was resumed
    Completed, // Timer completed normally
    Cancelled, // Timer was cancelled by user
}

/// Timer event record
#[derive(Debug, Clone)]
pub struct TimerEvent {
    pub timer_id: u64,              // Unique ID for the timer
    pub event_type: TimerEventType, // Type of event
    pub timestamp: SystemTime,      // When the event occurred
    pub timer_name: Option<String>, // Optional timer name
}

/// Timer configuration
#[derive(Clone, Default)]
pub struct TimerConfig {
    pub duration_ms: u64,          // Timer duration in milliseconds
    pub name: Option<String>,      // Optional name for the timer
    pub repeat: bool,              // Whether the timer should repeat
    pub repeat_count: Option<u32>, // Number of times to repeat (None = infinite)
    pub callback: Option<Arc<dyn Fn(TimerEvent) + Send + Sync>>, // Optional callback for this specific timer
}

// Manually implement Debug for TimerConfig to skip the callback field
impl fmt::Debug for TimerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimerConfig")
            .field("duration_ms", &self.duration_ms)
            .field("name", &self.name)
            .field("repeat", &self.repeat)
            .field("repeat_count", &self.repeat_count)
            .field(
                "callback",
                &if self.callback.is_some() {
                    "Some(Fn)"
                } else {
                    "None"
                },
            )
            .finish()
    }
}

/// Timer information
#[derive(Debug)]
pub struct Timer {
    pub id: u64,                           // Unique ID
    pub config: TimerConfig,               // Timer configuration
    pub state: TimerState,                 // Current state
    pub created_at: SystemTime,            // Creation time
    pub started_at: Option<Instant>,       // When the timer was last started/resumed
    pub elapsed_ms: u64,                   // Total elapsed time (across pauses)
    pub remaining_ms: u64,                 // Remaining time
    pub source_id: Option<glib::SourceId>, // GLib timer source ID
    pub repeat_remaining: Option<u32>,     // Remaining repeats
}

// Manually implement Clone for Timer since glib::SourceId doesn't implement Clone
impl Clone for Timer {
    fn clone(&self) -> Self {
        Timer {
            id: self.id,
            config: self.config.clone(),
            state: self.state.clone(),
            created_at: self.created_at,
            started_at: self.started_at,
            elapsed_ms: self.elapsed_ms,
            remaining_ms: self.remaining_ms,
            source_id: None, // We can't clone the source_id, so we set it to None
            repeat_remaining: self.repeat_remaining,
        }
    }
}

// Global timer handler function used with glib::timeout_add_once
// Modified to not return ControlFlow (since timeout_add_once handles this)
fn handle_timer_completion(timer_mgr: Arc<Mutex<TimerManager>>, timer_id: u64) {
    if let Ok(mut manager) = timer_mgr.lock() {
        manager.handle_timer_completion(timer_id);
    } else {
        error!("Failed to lock timer manager in completion handler");
    }
}

/// Timer manager for handling multiple timers
pub struct TimerManager {
    timers: HashMap<u64, Timer>,
    timer_events: Vec<TimerEvent>,
    next_timer_id: u64,
    pipeline_state: RecorderState,
    global_callback: Option<Arc<dyn Fn(TimerEvent) + Send + Sync>>,
}

impl TimerManager {
    /// Create a new timer manager
    pub fn new() -> Self {
        TimerManager {
            timers: HashMap::new(),
            timer_events: Vec::new(),
            next_timer_id: 1,
            pipeline_state: RecorderState::Ready,
            global_callback: None,
        }
    }

    /// Set global callback for all timer events
    pub fn set_callback<F>(&mut self, callback: F)
    where
        F: Fn(TimerEvent) + Send + Sync + 'static,
    {
        self.global_callback = Some(Arc::new(callback));
    }

    /// Check if there's an active timer (Ready, Running, or Paused)
    fn has_active_timer(&self) -> Option<u64> {
        for (id, timer) in &self.timers {
            if matches!(
                timer.state,
                TimerState::Ready | TimerState::Running | TimerState::Paused
            ) {
                return Some(*id);
            }
        }
        None
    }

    /// Create a new timer
    pub fn create_timer(
        &mut self,
        config: TimerConfig,
        timer_manager: Arc<Mutex<TimerManager>>,
    ) -> Result<u64> {
        // 验证配置有效性
        if config.duration_ms == 0 {
            warn!("Creating timer with zero duration");
        }

        // Check if there's an active timer
        if let Some(active_id) = self.has_active_timer() {
            debug!(
                "Cancelling existing active timer {} before creating new one",
                active_id
            );
            self.cancel_timer(active_id)?;
        }

        let timer_id = self.next_timer_id;
        self.next_timer_id += 1;

        let timer_name = config
            .name
            .clone()
            .unwrap_or_else(|| format!("Timer_{}", timer_id));

        let timer = Timer {
            id: timer_id,
            config: config.clone(),
            state: TimerState::Ready,
            created_at: SystemTime::now(),
            started_at: None,
            elapsed_ms: 0,
            remaining_ms: config.duration_ms,
            source_id: None,
            repeat_remaining: config.repeat_count,
        };

        debug!(
            "Creating timer id={} name={}: {:?}",
            timer_id, timer_name, timer
        );
        self.timers.insert(timer_id, timer);

        // Record event
        self.record_event(timer_id, TimerEventType::Created);

        // Start timer immediately if pipeline is playing
        if self.pipeline_state == RecorderState::Recording {
            debug!(
                "Pipeline is playing, starting timer {} immediately",
                timer_id
            );
            match self.start_timer(timer_manager) {
                Ok(_) => debug!("Timer {} started successfully", timer_id),
                Err(e) => error!("Failed to start timer {}: {}", timer_id, e),
            }
        } else {
            debug!(
                "Pipeline is not playing, timer {} will start when pipeline resumes",
                timer_id
            );
        }

        Ok(timer_id)
    }

    /// Start a timer
    pub fn start_timer(&mut self, timer_manager: Arc<Mutex<TimerManager>>) -> Result<()> {
        // We need to extract the timer_id first
        let timer_id = match self.has_active_timer() {
            Some(id) => id,
            None => {
                debug!("No active timer found to start");
                return Ok(());
            }
        };

        // First check if the timer exists
        if !self.timers.contains_key(&timer_id) {
            let err = RecorderError::ConfigError(format!("Timer {} not found", timer_id));
            error!("{}", err);
            return Err(err);
        }

        // Only start if pipeline is playing
        if self.pipeline_state != RecorderState::Recording {
            debug!("Pipeline is not playing, cannot start timer {}", timer_id);
            return Ok(());
        }

        // Check timer state before trying to start
        {
            let timer = self.timers.get(&timer_id).unwrap();
            if timer.state != TimerState::Ready && timer.state != TimerState::Paused {
                debug!("Timer {} is already in state {:?}", timer_id, timer.state);
                return Ok(());
            }
        }

        debug!("Starting timer {}", timer_id);

        // Prepare what we need for the timer
        let remaining_ms;

        {
            // Get a mutable reference to the timer and update its state
            let timer = self.timers.get_mut(&timer_id).unwrap();

            // Set start time
            timer.started_at = Some(Instant::now());

            // Ensure the remaining_ms is valid
            if timer.remaining_ms == 0 {
                warn!(
                    "Timer {} has remaining_ms=0, resetting to config duration",
                    timer_id
                );
                timer.remaining_ms = timer.config.duration_ms;
            }

            remaining_ms = timer.remaining_ms;
        }

        // Clone timer_id for the closure
        let timer_id_clone = timer_id;
        let timer_manager_clone = timer_manager.clone();

        // Create GLib timeout using timeout_add_once instead of timeout_add_local
        let source_id = glib::timeout_add_once(Duration::from_millis(remaining_ms), move || {
            handle_timer_completion(timer_manager_clone.clone(), timer_id_clone);
        });

        // Update the timer with the source_id
        {
            let timer = self.timers.get_mut(&timer_id).unwrap();
            timer.source_id = Some(source_id);
            timer.state = TimerState::Running;
        }

        // Record event
        self.record_event(timer_id, TimerEventType::Started);

        debug!(
            "Timer {} started successfully with remaining_ms={}",
            timer_id, remaining_ms
        );
        Ok(())
    }

    /// Pause a timer
    pub fn pause_timer(&mut self, timer_id: u64) -> Result<()> {
        // First check if the timer exists
        if !self.timers.contains_key(&timer_id) {
            let err = RecorderError::ConfigError(format!("Timer {} not found", timer_id));
            error!("{}", err);
            return Err(err);
        }

        // Check if the timer is running
        {
            let timer = self.timers.get(&timer_id).unwrap();
            if timer.state != TimerState::Running {
                debug!(
                    "Timer {} is not running (state={:?}), cannot pause",
                    timer_id, timer.state
                );
                return Ok(());
            }
        }

        // Handle the timer
        let remaining_ms;

        {
            let timer = self.timers.get_mut(&timer_id).unwrap();
            debug!("Pausing timer {}", timer_id);

            // Remove the GLib source
            if let Some(source_id) = timer.source_id.take() {
                source_id.remove();
            }

            // Calculate elapsed time and update remaining time
            if let Some(started_at) = timer.started_at {
                let elapsed = started_at.elapsed().as_millis() as u64;
                timer.elapsed_ms += elapsed;

                let old_remaining = timer.remaining_ms;
                timer.remaining_ms = timer.remaining_ms.saturating_sub(elapsed);
                remaining_ms = timer.remaining_ms;

                // Check if the timer should have completed
                if timer.remaining_ms == 0 && old_remaining > 0 {
                    debug!(
                        "Timer {} would have completed during pause, completing now",
                        timer_id
                    );
                    timer.state = TimerState::Completed;

                    // Save repeat info before dropping the borrow
                    let should_repeat = timer.config.repeat;
                    let repeat_count = timer.repeat_remaining;
                    let config_duration = timer.config.duration_ms;

                    // End the borrow with a scope block instead of drop()
                    {
                        // This scope ends the mutable borrow
                    }

                    // Record completion event
                    self.record_event(timer_id, TimerEventType::Completed);

                    // Handle repeating timers
                    if should_repeat {
                        let should_restart = match repeat_count {
                            Some(count) if count > 0 => {
                                // Decrement count
                                let timer = self.timers.get_mut(&timer_id).unwrap();
                                if let Some(count) = timer.repeat_remaining.as_mut() {
                                    *count -= 1;
                                }
                                true
                            }
                            None => true, // Infinite repeats
                            _ => false,
                        };

                        if should_restart {
                            debug!(
                                "Restarting repeating timer {} after completion during pause",
                                timer_id
                            );
                            let timer = self.timers.get_mut(&timer_id).unwrap();
                            timer.state = TimerState::Paused; // Will be resumed when pipeline is playing
                            timer.remaining_ms = config_duration;
                            timer.elapsed_ms = 0;
                        }
                    }

                    return Ok(());
                }
            } else {
                remaining_ms = timer.remaining_ms;
            }

            timer.started_at = None;
            timer.state = TimerState::Paused;
        }

        // Record event
        self.record_event(timer_id, TimerEventType::Paused);
        debug!(
            "Timer {} paused with remaining_ms={}",
            timer_id, remaining_ms
        );

        Ok(())
    }

    /// Resume a paused timer
    pub fn resume_timer(
        &mut self,
        timer_id: u64,
        timer_manager: Arc<Mutex<TimerManager>>,
    ) -> Result<()> {
        // First check if the timer exists
        if !self.timers.contains_key(&timer_id) {
            let err = RecorderError::ConfigError(format!("Timer {} not found", timer_id));
            error!("{}", err);
            return Err(err);
        }

        // Only resume if pipeline is playing
        if self.pipeline_state != RecorderState::Recording {
            debug!("Pipeline is not playing, cannot resume timer {}", timer_id);
            return Ok(());
        }

        // Check if the timer is paused and has remaining time
        let (can_resume, remaining_ms) = {
            let timer = self.timers.get(&timer_id).unwrap();
            if timer.state != TimerState::Paused {
                debug!(
                    "Timer {} is not paused (state={:?}), cannot resume",
                    timer_id, timer.state
                );
                return Ok(());
            }

            // Check if there's any time remaining
            let can_resume = timer.remaining_ms > 0;
            if !can_resume {
                debug!("Timer {} has no remaining time, cannot resume", timer_id);
            }
            (can_resume, timer.remaining_ms)
        };

        // Early return if we can't resume
        if !can_resume {
            // Move to completed state
            {
                let timer = self.timers.get_mut(&timer_id).unwrap();
                timer.state = TimerState::Completed;
            }
            self.record_event(timer_id, TimerEventType::Completed);
            return Ok(());
        }

        debug!("Resuming timer {}", timer_id);

        // Clone timer_id for the closure
        let timer_id_clone = timer_id;
        let timer_manager_clone = timer_manager.clone();

        // Create GLib timeout using timeout_add_once instead of timeout_add_local
        let source_id = glib::timeout_add_once(Duration::from_millis(remaining_ms), move || {
            handle_timer_completion(timer_manager_clone.clone(), timer_id_clone);
        });

        // Update the timer state
        {
            let timer = self.timers.get_mut(&timer_id).unwrap();
            // Set start time
            timer.started_at = Some(Instant::now());
            timer.source_id = Some(source_id);
            timer.state = TimerState::Running;
        }

        // Record event
        self.record_event(timer_id, TimerEventType::Resumed);
        debug!(
            "Timer {} resumed with remaining_ms={}",
            timer_id, remaining_ms
        );

        Ok(())
    }

    /// Cancel a timer
    pub fn cancel_timer(&mut self, timer_id: u64) -> Result<()> {
        // First check if the timer exists
        if !self.timers.contains_key(&timer_id) {
            let err = RecorderError::ConfigError(format!("Timer {} not found", timer_id));
            error!("{}", err);
            return Err(err);
        }

        // Check if the timer can be cancelled and get its state
        let can_cancel;
        let final_elapsed_ms;

        {
            let timer = self.timers.get(&timer_id).unwrap();
            can_cancel = timer.state == TimerState::Running
                || timer.state == TimerState::Paused
                || timer.state == TimerState::Ready;

            if !can_cancel {
                debug!(
                    "Timer {} is already in state {:?}, cannot cancel",
                    timer_id, timer.state
                );
                return Ok(());
            }
        }

        {
            let timer = self.timers.get_mut(&timer_id).unwrap();
            debug!("Cancelling timer {}", timer_id);

            // Remove the GLib source if running
            if let Some(source_id) = timer.source_id.take() {
                source_id.remove();
            }

            // Update elapsed time if running
            if timer.state == TimerState::Running {
                if let Some(started_at) = timer.started_at {
                    let elapsed = started_at.elapsed().as_millis() as u64;
                    timer.elapsed_ms += elapsed;
                }
            }

            timer.started_at = None;
            timer.state = TimerState::Cancelled;
            final_elapsed_ms = timer.elapsed_ms;
        }

        // Record event
        self.record_event(timer_id, TimerEventType::Cancelled);
        debug!(
            "Timer {} cancelled after elapsed_ms={}",
            timer_id, final_elapsed_ms
        );

        Ok(())
    }

    /// Handle pipeline state changes
    pub fn handle_pipeline_state_change(
        &mut self,
        new_state: RecorderState,
        timer_manager: Arc<Mutex<TimerManager>>,
    ) {
        let old_state = self.pipeline_state.clone(); // Clone to fix move error
        self.pipeline_state = new_state.clone();

        debug!(
            "Pipeline state changed from {:?} to {:?}",
            old_state, new_state
        );

        match new_state {
            RecorderState::Recording => {
                // 修改这里：无论之前的状态如何，都启动所有Ready状态的定时器
                debug!("Starting all ready timers");
                let ready_timers = self.get_timers_by_state(TimerState::Ready);
                let ready_count = ready_timers.len();

                if !ready_timers.is_empty() {
                    match self.start_timer(timer_manager.clone()) {
                        Ok(_) => debug!("Started timer"),
                        Err(e) => error!("Failed to start timer: {}", e),
                    }
                }
                debug!("Started {} ready timers", ready_count);

                // 然后再恢复所有被暂停的定时器（如果从Paused状态恢复）
                if old_state == RecorderState::Paused {
                    debug!("Resuming all paused timers");
                    let paused_timers = self.get_timers_by_state(TimerState::Paused);
                    let paused_count = paused_timers.len();

                    for timer_id in paused_timers {
                        match self.resume_timer(timer_id, timer_manager.clone()) {
                            Ok(_) => debug!("Resumed timer {}", timer_id),
                            Err(e) => error!("Failed to resume timer {}: {}", timer_id, e),
                        }
                    }
                    debug!("Resumed {} paused timers", paused_count);
                }
            }
            RecorderState::Paused => {
                // Pause all running timers
                debug!("Pausing all running timers");
                let running_timers = self.get_timers_by_state(TimerState::Running);
                let running_count = running_timers.len();

                for timer_id in running_timers {
                    match self.pause_timer(timer_id) {
                        Ok(_) => debug!("Paused timer {}", timer_id),
                        Err(e) => error!("Failed to pause timer {}: {}", timer_id, e),
                    }
                }
                debug!("Paused {} running timers", running_count);
            }
            RecorderState::Stopped => {
                // Cancel all timers
                debug!("Cancelling all active timers");
                let active_timers = self.get_active_timer_ids();
                let active_count = active_timers.len();

                for timer_id in active_timers {
                    match self.cancel_timer(timer_id) {
                        Ok(_) => debug!("Cancelled timer {}", timer_id),
                        Err(e) => error!("Failed to cancel timer {}: {}", timer_id, e),
                    }
                }
                debug!("Cancelled {} active timers", active_count);
            }
            _ => {}
        }
    }

    /// Get timer IDs by state
    fn get_timers_by_state(&self, state: TimerState) -> Vec<u64> {
        self.timers
            .iter()
            .filter_map(|(id, timer)| {
                if timer.state == state {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get active timer IDs (running or paused)
    fn get_active_timer_ids(&self) -> Vec<u64> {
        self.timers
            .iter()
            .filter_map(|(id, timer)| {
                if timer.state == TimerState::Running
                    || timer.state == TimerState::Paused
                    || timer.state == TimerState::Ready
                {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Handle timer completion
    fn handle_timer_completion(&mut self, timer_id: u64) {
        // First check if the timer exists
        if !self.timers.contains_key(&timer_id) {
            error!("Timer {} not found during completion handling", timer_id);
            return;
        }

        // Extract info we need to handle completion
        let should_repeat;
        let repeat_remaining;
        let config_duration;
        let elapsed_ms;

        {
            // Get the timer and update its state
            let timer = self.timers.get_mut(&timer_id).unwrap();
            debug!("Timer {} completed", timer_id);

            // Update elapsed time
            if let Some(started_at) = timer.started_at {
                let elapsed = started_at.elapsed().as_millis() as u64;
                timer.elapsed_ms += elapsed;
            }

            timer.started_at = None;
            timer.remaining_ms = 0;
            timer.source_id = None;
            timer.state = TimerState::Completed;

            // Save info we need after dropping the borrow
            should_repeat = timer.config.repeat;
            repeat_remaining = timer.repeat_remaining;
            config_duration = timer.config.duration_ms;
            elapsed_ms = timer.elapsed_ms;
        }

        // Record completion event
        self.record_event(timer_id, TimerEventType::Completed);
        debug!(
            "Timer {} completed after elapsed_ms={}",
            timer_id, elapsed_ms
        );

        // Handle repeating timers
        if should_repeat {
            debug!(
                "Timer {} is set to repeat (remaining: {:?})",
                timer_id, repeat_remaining
            );

            let should_restart = match repeat_remaining {
                Some(count) if count > 0 => {
                    // Decrement count
                    if let Some(timer) = self.timers.get_mut(&timer_id) {
                        if let Some(count) = timer.repeat_remaining.as_mut() {
                            *count -= 1;
                            debug!("Timer {} has {} repeats remaining", timer_id, *count);
                        }
                    }
                    true
                }
                Some(0) => {
                    debug!("Timer {} has no repeats remaining", timer_id);
                    false
                }
                None => {
                    debug!("Timer {} has infinite repeats", timer_id);
                    true
                }
                _ => false,
            };

            if should_restart {
                debug!("Restarting repeating timer {}", timer_id);

                // Reset the timer
                {
                    let timer = self.timers.get_mut(&timer_id).unwrap();
                    timer.state = TimerState::Ready;
                    timer.remaining_ms = config_duration;
                    timer.elapsed_ms = 0;
                }

                // The timer will be restarted by the regular timer management process
                // when the pipeline is playing
            }
        }
    }

    /// Record a timer event
    fn record_event(&mut self, timer_id: u64, event_type: TimerEventType) {
        let timer_name = self
            .timers
            .get(&timer_id)
            .and_then(|t| t.config.name.clone());

        let event = TimerEvent {
            timer_id,
            event_type: event_type.clone(),
            timestamp: SystemTime::now(),
            timer_name: timer_name.clone(),
        };

        debug!("Timer event: timer_id={}, type={:?}", timer_id, event_type);
        self.timer_events.push(event.clone());

        // Call the global callback if set
        if let Some(cb) = &self.global_callback {
            cb(event.clone());
        }

        // Call the timer-specific callback if set
        if let Some(timer) = self.timers.get(&timer_id) {
            if let Some(cb) = &timer.config.callback {
                cb(event);
            }
        }
    }

    /// Get all timer events
    pub fn get_timer_events(&self) -> &Vec<TimerEvent> {
        &self.timer_events
    }

    /// Get active timers (running or paused)
    pub fn get_active_timers(&self) -> Vec<Timer> {
        self.timers
            .values()
            .filter(|t| t.state == TimerState::Running || t.state == TimerState::Paused)
            .cloned()
            .collect()
    }

    /// Get a specific timer
    pub fn get_timer(&self, timer_id: u64) -> Option<&Timer> {
        self.timers.get(&timer_id)
    }

    /// Get all timers
    pub fn get_all_timers(&self) -> Vec<Timer> {
        self.timers.values().cloned().collect()
    }
}
