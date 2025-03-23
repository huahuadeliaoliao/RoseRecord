// lib.rs
use glib::{ControlFlow, MainLoop};
use gst::prelude::*;
use gstreamer as gst;
use log::{debug, error, info, trace, warn};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

mod errors;
pub use errors::{PluginError, PluginResult, RecorderError, Result};

pub mod plugins;

mod timer;
pub use timer::{Timer, TimerConfig, TimerEvent, TimerEventType, TimerState};

/// Recording configuration options
#[derive(Debug, Clone)]
pub struct RecorderConfig {
    /// Output file path
    pub output_file: String,
    /// Audio format
    pub format: AudioFormat,
    /// Bitrate (kbps)
    pub bitrate: i32,
    /// List of enabled plugins
    pub enabled_plugins: Vec<PluginType>,
}

/// Supported audio formats
#[derive(Debug, Clone, PartialEq)]
pub enum AudioFormat {
    MP3,
    WAV,
    FLAC,
    OGG,
}

/// Available plugin types
#[derive(Debug, Clone, PartialEq)]
pub enum PluginType {
    NoiseReduction,
    SilenceRemoval,
}

/// Plugin trait
pub trait RecorderPlugin {
    /// Plugin name
    fn name(&self) -> &str;

    /// Plugin type
    fn plugin_type(&self) -> PluginType;

    /// Add plugin to GStreamer pipeline
    fn add_to_pipeline(
        &self,
        pipeline: &gst::Pipeline,
        previous_element: &gst::Element,
    ) -> Result<gst::Element>;

    /// Configure plugin
    fn configure(&mut self, config: HashMap<String, String>) -> PluginResult<()>;

    /// Handle GStreamer messages
    fn handle_message(&self, msg: &gst::Message) -> PluginResult<()>;
}

/// Recorder state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecorderState {
    Ready,
    Recording,
    Paused,
    Stopped,
}

/// Recorder callback events
pub enum RecorderEvent {
    StateChanged(RecorderState),
    DurationUpdate(u64),
    LevelUpdate(f64),
    TimerEvent(TimerEvent), // Added timer event
    Error(String),
}

/// Recorder main structure
pub struct Recorder {
    config: RecorderConfig,
    pipeline: Option<Arc<gst::Pipeline>>, // Changed to Arc
    plugins: Vec<Box<dyn RecorderPlugin>>,
    state: Arc<Mutex<RecorderState>>,
    callback: Option<Arc<dyn Fn(RecorderEvent) + Send + Sync>>,
    timer_manager: Arc<Mutex<timer::TimerManager>>,
    main_loop: Option<Arc<MainLoop>>,
    main_loop_thread: Option<thread::JoinHandle<()>>,
}

impl Recorder {
    /// Create a new recorder instance
    pub fn new(config: RecorderConfig) -> Self {
        // Validate configuration validity
        Self::validate_config(&config);

        // Initialize GStreamer
        if gst::init().is_err() {
            error!("Cannot initialize GStreamer, some features may not work properly");
        }

        // Create timer manager, ensure thread safety
        let timer_manager = Arc::new(Mutex::new(timer::TimerManager::new()));

        Recorder {
            config,
            pipeline: None,
            plugins: Vec::new(),
            state: Arc::new(Mutex::new(RecorderState::Ready)),
            callback: None,
            timer_manager,
            main_loop: None,
            main_loop_thread: None,
        }
    }

    // Validate configuration
    fn validate_config(config: &RecorderConfig) {
        if config.output_file.is_empty() {
            warn!("Recording config: Output file path is empty");
        }

        if config.bitrate <= 0 {
            warn!(
                "Recording config: Bitrate {} is invalid, will use default value",
                config.bitrate
            );
        }

        match config.format {
            AudioFormat::MP3 => {}
            AudioFormat::WAV => {}
            AudioFormat::FLAC => {}
            AudioFormat::OGG => {}
        }

        if config.enabled_plugins.is_empty() {
            info!("Recording config: No plugins enabled");
        } else {
            debug!(
                "Recording config: {} plugins enabled",
                config.enabled_plugins.len()
            );
        }
    }

    /// Register event callback
    pub fn set_callback<F>(&mut self, callback: F)
    where
        F: Fn(RecorderEvent) + Send + Sync + 'static,
    {
        let callback_arc = Arc::new(callback);
        self.callback = Some(callback_arc.clone());

        // Set timer event forwarding
        let cb = callback_arc.clone();
        if let Ok(mut timer_manager) = self.timer_manager.lock() {
            timer_manager.set_callback(move |timer_event| {
                cb(RecorderEvent::TimerEvent(timer_event));
            });
        } else {
            error!("Cannot lock timer manager to set callback");
        }
    }

    /// Trigger event callback
    fn emit_event(&self, event: RecorderEvent) {
        if let Some(callback) = &self.callback {
            callback(event);
        }
    }

    /// Register plugin
    pub fn register_plugin(&mut self, plugin: Box<dyn RecorderPlugin>) {
        debug!("Registering plugin: {}", plugin.name());
        self.plugins.push(plugin);
    }

    /// Build GStreamer pipeline
    fn build_pipeline(&mut self) -> Result<()> {
        // Create pipeline
        let pipeline = gst::Pipeline::new();

        // Create basic elements
        let source = gst::ElementFactory::make("autoaudiosrc")
            .build()
            .map_err(|e| {
                RecorderError::ElementCreationError(format!("Cannot create audio source: {}", e))
            })?;

        let convert1 = gst::ElementFactory::make("audioconvert")
            .build()
            .map_err(|e| {
                RecorderError::ElementCreationError(format!("Cannot create audio converter: {}", e))
            })?;

        let resample1 = gst::ElementFactory::make("audioresample")
            .build()
            .map_err(|e| {
                RecorderError::ElementCreationError(format!("Cannot create audio resampler: {}", e))
            })?;

        // Add to pipeline and link basic elements
        pipeline
            .add_many([&source, &convert1, &resample1])
            .map_err(|e| {
                RecorderError::PipelineError(format!("Cannot add elements to pipeline: {}", e))
            })?;

        gst::Element::link_many([&source, &convert1, &resample1])
            .map_err(|e| RecorderError::ElementLinkError { source: e })?;

        // Last processing element, starting from resample1
        let mut last_element = resample1.clone();

        // Add enabled plugins based on configuration
        for plugin in &self.plugins {
            if self.config.enabled_plugins.contains(&plugin.plugin_type()) {
                debug!("Adding plugin to pipeline: {}", plugin.name());
                match plugin.add_to_pipeline(&pipeline, &last_element) {
                    Ok(new_element) => {
                        // Update last element
                        last_element = new_element;
                    }
                    Err(e) => {
                        error!("Failed to add plugin {}: {}", plugin.name(), e);
                        return Err(RecorderError::PluginError(format!(
                            "Failed to add plugin {}: {}",
                            plugin.name(),
                            e
                        )));
                    }
                }
            }
        }

        // Create encoder and output elements
        let (encoder, sink) = self.create_encoder_and_sink()?;

        // Add encoder and output to pipeline
        pipeline.add_many([&encoder, &sink]).map_err(|e| {
            RecorderError::PipelineError(format!(
                "Cannot add encoder and output to pipeline: {}",
                e
            ))
        })?;

        // Link last processing element to encoder, then to output
        last_element
            .link(&encoder)
            .map_err(|e| RecorderError::ElementLinkError { source: e })?;
        encoder
            .link(&sink)
            .map_err(|e| RecorderError::ElementLinkError { source: e })?;

        // Set pipeline reference, wrapped with Arc
        self.pipeline = Some(Arc::new(pipeline));
        debug!("GStreamer pipeline build complete");

        Ok(())
    }

    /// Create encoder and output elements based on configuration
    fn create_encoder_and_sink(&self) -> Result<(gst::Element, gst::Element)> {
        let (encoder_factory, encoder_properties) = match self.config.format {
            AudioFormat::MP3 => ("lamemp3enc", vec![("bitrate", &self.config.bitrate)]),
            AudioFormat::WAV => ("wavenc", vec![]),
            AudioFormat::FLAC => ("flacenc", vec![]),
            AudioFormat::OGG => ("vorbisenc", vec![("bitrate", &self.config.bitrate)]),
        };

        let encoder = gst::ElementFactory::make(encoder_factory)
            .build()
            .map_err(|e| {
                RecorderError::ElementCreationError(format!(
                    "Cannot create encoder {}: {}",
                    encoder_factory, e
                ))
            })?;

        // Set encoder properties
        for (prop, value) in encoder_properties {
            encoder.set_property(prop, value);
        }

        // Create file output
        let sink = gst::ElementFactory::make("filesink").build().map_err(|e| {
            RecorderError::ElementCreationError(format!("Cannot create file output: {}", e))
        })?;
        sink.set_property("location", &self.config.output_file);

        debug!(
            "Created encoder {} and file output {}",
            encoder_factory, self.config.output_file
        );
        Ok((encoder, sink))
    }

    /// Start recording
    pub fn start(&mut self) -> Result<()> {
        // Check current state
        let current_state = self.state();
        if current_state == RecorderState::Recording {
            debug!("Recorder is already recording");
            return Ok(());
        }

        // Build pipeline if not already built
        if self.pipeline.is_none() {
            self.build_pipeline()?;
        }

        // Get Arc<Pipeline> clone
        let pipeline_arc = self
            .pipeline
            .as_ref()
            .ok_or_else(|| RecorderError::PipelineError("Pipeline not initialized".to_string()))?
            .clone();

        // Get pipeline message bus
        let bus = pipeline_arc
            .bus()
            .ok_or_else(|| RecorderError::BusError("Cannot get bus".to_string()))?;

        // Clone state reference for use in callback
        let state = self.state.clone();
        let callback = self.callback.clone();
        let timer_manager = self.timer_manager.clone();
        let pipeline_arc_for_closure = pipeline_arc.clone();

        // Set bus watcher
        let _bus_watch = bus
            .add_watch(move |_, msg| {
                use gst::MessageView;

                match msg.view() {
                    MessageView::Element(element_msg) => {
                        // Handle element messages
                        trace!("Received element message: {:?}", element_msg);
                    }
                    MessageView::Eos(..) => {
                        info!("Received EOS message");
                        // Set state to stopped
                        if let Ok(mut guard) = state.lock() {
                            *guard = RecorderState::Stopped;
                        } else {
                            error!("Cannot lock state mutex");
                        }

                        if let Ok(mut timer_mgr) = timer_manager.lock() {
                            timer_mgr.handle_pipeline_state_change(
                                RecorderState::Stopped,
                                timer_manager.clone(),
                            );
                        } else {
                            error!("Cannot lock timer manager mutex");
                        }

                        if let Some(cb) = &callback {
                            cb(RecorderEvent::StateChanged(RecorderState::Stopped));
                        }
                    }
                    MessageView::Error(err) => {
                        error!(
                            "Error from {:?}: {} ({:?})",
                            err.src().map(|s| s.path_string()),
                            err.error(),
                            err.debug()
                        );
                        // Send error event
                        if let Some(cb) = &callback {
                            cb(RecorderEvent::Error(format!(
                                "GStreamer error: {}",
                                err.error()
                            )));
                        }
                    }
                    MessageView::Warning(warn_msg) => {
                        warn!(
                            "Warning from {:?}: {} ({:?})",
                            warn_msg.src().map(|s| s.path_string()),
                            warn_msg.error(),
                            warn_msg.debug()
                        );
                    }
                    MessageView::StateChanged(state_changed) => {
                        if let Some(src) = state_changed.src() {
                            if pipeline_arc_for_closure
                                .upcast_ref::<gst::Object>()
                                .as_ptr()
                                == src.as_ptr()
                            {
                                trace!(
                                    "Pipeline state change: {:?} -> {:?} (pending: {:?})",
                                    state_changed.old(),
                                    state_changed.current(),
                                    state_changed.pending()
                                );
                            }
                        }
                    }
                    _ => (),
                };

                ControlFlow::Continue
            })
            .map_err(|e| RecorderError::BusError(format!("Cannot add bus watcher: {}", e)))?;

        // Start pipeline
        match pipeline_arc.set_state(gst::State::Playing) {
            Ok(gst::StateChangeSuccess::Success) => {
                debug!("Pipeline state set to Playing successfully");
            }
            Ok(gst::StateChangeSuccess::Async) => {
                debug!("Pipeline state set to Playing asynchronously, waiting for state change...");
            }
            Ok(gst::StateChangeSuccess::NoPreroll) => {
                debug!("Pipeline state set to Playing (NoPreroll)");
            }
            Err(e) => {
                let err = RecorderError::StateError(format!("Cannot start pipeline: {}", e));
                error!("{}", err);
                return Err(err);
            }
        }

        // Update state
        {
            let mut state_guard = self
                .state
                .lock()
                .map_err(|_| RecorderError::StateError("Cannot lock state mutex".to_string()))?;
            *state_guard = RecorderState::Recording;
        }

        {
            let mut timer_manager_guard = self.timer_manager.lock().map_err(|_| {
                RecorderError::StateError("Cannot lock timer manager mutex".to_string())
            })?;
            timer_manager_guard
                .handle_pipeline_state_change(RecorderState::Recording, self.timer_manager.clone());
        }

        info!("Recording started");
        self.emit_event(RecorderEvent::StateChanged(RecorderState::Recording));

        // Create and start GLib main loop
        let main_loop = MainLoop::new(None, false);
        let main_loop_arc = Arc::new(main_loop);

        // Run GLib main loop in a new thread
        let main_loop_run = main_loop_arc.clone();
        let main_thread = thread::spawn(move || {
            debug!("GLib main loop started");
            main_loop_run.run();
            debug!("GLib main loop stopped");
        });

        // Store main loop reference for later stopping
        self.main_loop = Some(main_loop_arc);
        self.main_loop_thread = Some(main_thread);

        Ok(())
    }

    /// Pause recording
    pub fn pause(&mut self) -> Result<()> {
        // Check current state
        let current_state = self.state();
        if current_state != RecorderState::Recording {
            debug!(
                "Recorder is not in recording state, current state: {:?}",
                current_state
            );
            return Ok(());
        }

        if let Some(pipeline_arc) = &self.pipeline {
            // Use cloned Arc
            let pipeline = pipeline_arc.clone();
            match pipeline.set_state(gst::State::Paused) {
                Ok(_) => {
                    debug!("Pipeline state set to Paused successfully");
                }
                Err(e) => {
                    let err = RecorderError::StateError(format!("Cannot pause pipeline: {}", e));
                    error!("{}", err);
                    return Err(err);
                }
            }

            // Update state
            {
                let mut state_guard = self.state.lock().map_err(|_| {
                    RecorderError::StateError("Cannot lock state mutex".to_string())
                })?;
                *state_guard = RecorderState::Paused;
            }

            {
                let mut timer_manager_guard = self.timer_manager.lock().map_err(|_| {
                    RecorderError::StateError("Cannot lock timer manager mutex".to_string())
                })?;
                timer_manager_guard.handle_pipeline_state_change(
                    RecorderState::Paused,
                    self.timer_manager.clone(),
                );
            }

            info!("Recording paused");
            self.emit_event(RecorderEvent::StateChanged(RecorderState::Paused));
        } else {
            let err = RecorderError::PipelineError("Pipeline not initialized".to_string());
            error!("{}", err);
            return Err(err);
        }
        Ok(())
    }

    /// Resume recording
    pub fn resume(&mut self) -> Result<()> {
        // Check current state
        let current_state = self.state();
        if current_state != RecorderState::Paused {
            debug!(
                "Recorder is not in paused state, current state: {:?}",
                current_state
            );
            return Ok(());
        }

        if let Some(pipeline_arc) = &self.pipeline {
            // Use cloned Arc
            let pipeline = pipeline_arc.clone();
            match pipeline.set_state(gst::State::Playing) {
                Ok(_) => {
                    debug!("Pipeline state set to Playing successfully");
                }
                Err(e) => {
                    let err = RecorderError::StateError(format!("Cannot resume pipeline: {}", e));
                    error!("{}", err);
                    return Err(err);
                }
            }

            // Update state
            {
                let mut state_guard = self.state.lock().map_err(|_| {
                    RecorderError::StateError("Cannot lock state mutex".to_string())
                })?;
                *state_guard = RecorderState::Recording;
            }

            {
                let mut timer_manager_guard = self.timer_manager.lock().map_err(|_| {
                    RecorderError::StateError("Cannot lock timer manager mutex".to_string())
                })?;
                timer_manager_guard.handle_pipeline_state_change(
                    RecorderState::Recording,
                    self.timer_manager.clone(),
                );
            }

            info!("Recording resumed");
            self.emit_event(RecorderEvent::StateChanged(RecorderState::Recording));
        } else {
            let err = RecorderError::PipelineError("Pipeline not initialized".to_string());
            error!("{}", err);
            return Err(err);
        }
        Ok(())
    }

    /// Stop recording
    pub fn stop(&mut self) -> Result<()> {
        // Check current state
        let current_state = self.state();
        if current_state == RecorderState::Stopped || current_state == RecorderState::Ready {
            debug!(
                "Recorder already stopped or not started, current state: {:?}",
                current_state
            );
            return Ok(());
        }

        if let Some(pipeline_arc) = &self.pipeline {
            let pipeline = pipeline_arc.clone();

            // Get pipeline bus
            let bus = pipeline
                .bus()
                .ok_or_else(|| RecorderError::BusError("Cannot get bus".to_string()))?;

            // First set pipeline to PAUSED state
            debug!("Setting pipeline to paused state");
            match pipeline.set_state(gst::State::Paused) {
                Ok(gst::StateChangeSuccess::Success) => {
                    debug!("Pipeline immediately set to PAUSED state successfully");
                }
                Ok(gst::StateChangeSuccess::Async) => {
                    debug!(
                        "Pipeline asynchronously set to PAUSED state, waiting for state change to complete"
                    );
                    // Use simplified method to wait for state change
                    debug!("Waiting for PAUSED state change to complete");
                    // Use while loop to process messages until we find the state change message we want
                    let mut found_state_change = false;
                    while !found_state_change {
                        if let Some(msg) = bus.timed_pop_filtered(
                            gst::ClockTime::from_seconds(5),
                            &[gst::MessageType::StateChanged],
                        ) {
                            if let gst::MessageView::StateChanged(state_changed) = msg.view() {
                                if let Some(src) = msg.src() {
                                    if src.as_ptr() == pipeline.upcast_ref::<gst::Object>().as_ptr()
                                        && state_changed.current() == gst::State::Paused
                                    {
                                        debug!("State has changed to PAUSED");
                                        found_state_change = true;
                                    }
                                }
                            }
                        } else {
                            // Timeout, continue anyway
                            debug!("Waiting for state change timed out, continuing process");
                            break;
                        }
                    }
                }
                Ok(gst::StateChangeSuccess::NoPreroll) => {
                    debug!("Pipeline set to PAUSED state (NoPreroll)");
                }
                Err(e) => {
                    let err = RecorderError::StateError(format!(
                        "Cannot set pipeline to PAUSED state: {}",
                        e
                    ));
                    error!("{}", err);
                    return Err(err);
                }
            }

            // Send EOS event
            debug!("Sending EOS event");
            if !pipeline.send_event(gst::event::Eos::new()) {
                warn!("Failed to send EOS event, continuing stop process");
            } else {
                // Listen for EOS message, use timeout to avoid infinite waiting
                debug!("Waiting for EOS message");
                let _ = bus.timed_pop_filtered(
                    gst::ClockTime::from_seconds(3), // Set 3 second timeout
                    &[gst::MessageType::Eos],
                );
            }

            // Finally set state to NULL
            debug!("Setting pipeline state to Null");
            match pipeline.set_state(gst::State::Null) {
                Ok(_) => {
                    debug!("Pipeline state set to NULL successfully");
                }
                Err(e) => {
                    let err = RecorderError::StateError(format!(
                        "Cannot set pipeline to NULL state: {}",
                        e
                    ));
                    error!("{}", err);
                    return Err(err);
                }
            }

            // Now stop GLib main loop (if still running)
            if let Some(main_loop) = &self.main_loop {
                debug!("Stopping GLib main loop");
                main_loop.quit();
            }

            // Wait for main loop thread to end
            if let Some(thread_handle) = self.main_loop_thread.take() {
                debug!("Waiting for GLib main loop thread to end");
                if thread_handle.join().is_err() {
                    error!("Error waiting for GLib main loop thread to end");
                }
            }

            // Update state
            {
                let mut state_guard = self.state.lock().map_err(|_| {
                    RecorderError::StateError("Cannot lock state mutex".to_string())
                })?;
                *state_guard = RecorderState::Stopped;
            }

            // Update timer state
            {
                let mut timer_manager_guard = self.timer_manager.lock().map_err(|_| {
                    RecorderError::StateError("Cannot lock timer manager mutex".to_string())
                })?;
                timer_manager_guard.handle_pipeline_state_change(
                    RecorderState::Stopped,
                    self.timer_manager.clone(),
                );
            }

            info!("Recording stopped");
            self.emit_event(RecorderEvent::StateChanged(RecorderState::Stopped));
        } else {
            let err = RecorderError::PipelineError("Pipeline not initialized".to_string());
            error!("{}", err);
            return Err(err);
        }

        Ok(())
    }

    /// Get current state
    pub fn state(&self) -> RecorderState {
        match self.state.lock() {
            Ok(guard) => (*guard).clone(),
            Err(_) => {
                error!("Cannot lock state mutex, returning default state Ready");
                RecorderState::Ready
            }
        }
    }

    /// Get recording duration (milliseconds)
    pub fn duration(&self) -> Option<u64> {
        if let Some(pipeline_arc) = &self.pipeline {
            // Use cloned Arc
            let pipeline = pipeline_arc.clone();
            let position = pipeline.query_position::<gst::ClockTime>();
            position.map(|p| p.mseconds())
        } else {
            None
        }
    }

    /// Create a new timer (only one active timer allowed)
    pub fn create_timer(&self, config: TimerConfig) -> Result<u64> {
        // Validate timer configuration
        if config.duration_ms == 0 {
            warn!("Creating timer: Duration is 0");
        }

        match self.timer_manager.lock() {
            Ok(mut timer_manager) => {
                // No need to explicitly check for existing active timers here, because
                // TimerManager::create_timer already includes this logic and will
                // automatically cancel any existing active timer

                // Pass self.timer_manager.clone() as the second parameter
                let timer_id = timer_manager.create_timer(config, self.timer_manager.clone())?;
                debug!("Timer created successfully, ID: {}", timer_id);
                Ok(timer_id)
            }
            Err(_) => {
                let err = RecorderError::StateError("Cannot lock timer manager mutex".to_string());
                error!("{}", err);
                Err(err)
            }
        }
    }

    /// Cancel timer
    pub fn cancel_timer(&self, timer_id: u64) -> Result<()> {
        match self.timer_manager.lock() {
            Ok(mut timer_manager) => timer_manager.cancel_timer(timer_id),
            Err(_) => {
                let err = RecorderError::StateError("Cannot lock timer manager mutex".to_string());
                error!("{}", err);
                Err(err)
            }
        }
    }

    /// Get all timer events
    pub fn get_timer_events(&self) -> Result<Vec<TimerEvent>> {
        match self.timer_manager.lock() {
            Ok(timer_manager) => {
                let events = timer_manager.get_timer_events().clone();
                Ok(events)
            }
            Err(_) => {
                let err = RecorderError::StateError("Cannot lock timer manager mutex".to_string());
                error!("{}", err);
                Err(err)
            }
        }
    }

    /// Get active timers
    pub fn get_active_timers(&self) -> Result<Vec<Timer>> {
        match self.timer_manager.lock() {
            Ok(timer_manager) => {
                let timers = timer_manager.get_active_timers();
                Ok(timers)
            }
            Err(_) => {
                let err = RecorderError::StateError("Cannot lock timer manager mutex".to_string());
                error!("{}", err);
                Err(err)
            }
        }
    }

    /// Get all timers
    pub fn get_all_timers(&self) -> Result<Vec<Timer>> {
        match self.timer_manager.lock() {
            Ok(timer_manager) => {
                let timers = timer_manager.get_all_timers();
                Ok(timers)
            }
            Err(_) => {
                let err = RecorderError::StateError("Cannot lock timer manager mutex".to_string());
                error!("{}", err);
                Err(err)
            }
        }
    }

    /// Get specific timer
    pub fn get_timer(&self, timer_id: u64) -> Result<Option<Timer>> {
        match self.timer_manager.lock() {
            Ok(timer_manager) => {
                let timer = timer_manager.get_timer(timer_id).cloned();
                Ok(timer)
            }
            Err(_) => {
                let err = RecorderError::StateError("Cannot lock timer manager mutex".to_string());
                error!("{}", err);
                Err(err)
            }
        }
    }
}

// Ensure resources are cleaned up in the destructor
impl Drop for Recorder {
    fn drop(&mut self) {
        debug!("Recorder is being destroyed, cleaning up resources...");

        // Ensure pipeline is stopped
        if let Some(pipeline_arc) = &self.pipeline {
            // Use cloned Arc
            let pipeline = pipeline_arc.clone();
            debug!("Stopping pipeline");
            if let Err(e) = pipeline.set_state(gst::State::Null) {
                error!("Failed to stop pipeline during destruction: {}", e);
            }
        }

        // Ensure GLib main loop is stopped
        if let Some(main_loop) = &self.main_loop {
            debug!("Stopping GLib main loop");
            main_loop.quit();
        }

        // Try to wait for main loop thread to end (with timeout to avoid potential blocking)
        if let Some(thread_handle) = self.main_loop_thread.take() {
            debug!("Waiting for GLib main loop thread to end");
            let thread_join_handle = std::thread::spawn(move || {
                if thread_handle.join().is_err() {
                    error!("Error waiting for GLib main loop thread to end");
                }
            });

            // Wait at most 1 second
            thread_join_handle.join().ok();
        }

        debug!("Recorder resource cleanup complete");
    }
}

// Default configuration
impl Default for RecorderConfig {
    fn default() -> Self {
        RecorderConfig {
            output_file: "recording.mp3".to_string(),
            format: AudioFormat::MP3,
            bitrate: 128,
            enabled_plugins: vec![],
        }
    }
}
