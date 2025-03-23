// plugins.rs
use std::collections::HashMap;

use gst::prelude::*;
use gstreamer as gst;
use log::{debug, error, trace, warn};

use crate::{PluginError, PluginResult, PluginType, RecorderError, RecorderPlugin, Result};

/// Noise Reduction plugin (RNNoise)
pub struct NoiseReductionPlugin {
    vad_threshold: f32,
}

impl NoiseReductionPlugin {
    pub fn new(vad_threshold: f32) -> Self {
        debug!(
            "Creating noise reduction plugin, threshold: {}",
            vad_threshold
        );
        NoiseReductionPlugin { vad_threshold }
    }
}

impl RecorderPlugin for NoiseReductionPlugin {
    fn name(&self) -> &str {
        "Noise Reduction"
    }

    fn plugin_type(&self) -> PluginType {
        PluginType::NoiseReduction
    }

    fn add_to_pipeline(
        &self,
        pipeline: &gst::Pipeline,
        previous_element: &gst::Element,
    ) -> Result<gst::Element> {
        // Create RNNoise element
        let rnnoise = gst::ElementFactory::make("audiornnoise")
            .build()
            .map_err(|_| {
                RecorderError::ElementCreationError(
                    "Cannot create audiornnoise element".to_string(),
                )
            })?;

        // Set voice activity detection threshold
        if self.vad_threshold > 0.0 {
            rnnoise.set_property("voice-activity-threshold", self.vad_threshold);
            debug!("Set noise reduction threshold: {}", self.vad_threshold);
        }

        // Create converter
        let convert = gst::ElementFactory::make("audioconvert")
            .build()
            .map_err(|_| {
                RecorderError::ElementCreationError("Cannot create audio converter".to_string())
            })?;

        // Add to pipeline
        pipeline.add_many([&rnnoise, &convert]).map_err(|_| {
            RecorderError::PipelineError(
                "Cannot add noise reduction elements to pipeline".to_string(),
            )
        })?;

        // Link elements
        previous_element.link(&rnnoise)?;
        rnnoise.link(&convert)?;

        debug!("Noise reduction plugin added to pipeline");
        // Return the last element in the chain
        Ok(convert)
    }

    fn configure(&mut self, config: HashMap<String, String>) -> PluginResult<()> {
        if let Some(threshold_str) = config.get("vad_threshold") {
            match threshold_str.parse::<f32>() {
                Ok(threshold) => {
                    self.vad_threshold = threshold;
                    debug!("Updated noise reduction threshold: {}", threshold);
                    Ok(())
                }
                Err(_) => {
                    let err = PluginError::ParameterError(
                        "Invalid voice activity threshold setting".to_string(),
                    );
                    error!("{}", err);
                    Err(err)
                }
            }
        } else {
            trace!("No noise reduction threshold configuration provided");
            Ok(())
        }
    }

    fn handle_message(&self, _msg: &gst::Message) -> PluginResult<()> {
        // RNNoise typically doesn't send messages that need special handling
        Ok(())
    }
}

/// Silence removal plugin
pub struct SilenceRemovalPlugin {
    threshold: i32,
    minimum_silence_time: u64,
    squash: bool,
    hysteresis: u64,
}

impl SilenceRemovalPlugin {
    pub fn new(threshold: i32, minimum_silence_time: u64) -> Self {
        debug!(
            "Creating silence removal plugin, threshold: {}, minimum silence time: {}",
            threshold, minimum_silence_time
        );
        SilenceRemovalPlugin {
            threshold,
            minimum_silence_time,
            squash: false,
            hysteresis: 480, // Default value from gst-inspect
        }
    }
}

impl RecorderPlugin for SilenceRemovalPlugin {
    fn name(&self) -> &str {
        "Silence Removal"
    }

    fn plugin_type(&self) -> PluginType {
        PluginType::SilenceRemoval
    }

    fn add_to_pipeline(
        &self,
        pipeline: &gst::Pipeline,
        previous_element: &gst::Element,
    ) -> Result<gst::Element> {
        // Create removesilence element
        let removesilence = gst::ElementFactory::make("removesilence")
            .build()
            .map_err(|_| {
                RecorderError::ElementCreationError(
                    "Cannot create removesilence element".to_string(),
                )
            })?;

        // Set properties
        removesilence.set_property("remove", true);
        removesilence.set_property("silent", false); // Enable bus message notifications
        removesilence.set_property("squash", self.squash);
        removesilence.set_property("hysteresis", self.hysteresis);

        if self.threshold != 0 {
            removesilence.set_property("threshold", self.threshold);
            debug!("Set silence removal threshold: {} dB", self.threshold);
        }

        if self.minimum_silence_time > 0 {
            removesilence.set_property("minimum-silence-time", self.minimum_silence_time);
            debug!("Set minimum silence time: {} ns", self.minimum_silence_time);
        }

        // Create valve element for flow control
        let valve = gst::ElementFactory::make("valve").build().map_err(|_| {
            RecorderError::ElementCreationError("Cannot create valve element".to_string())
        })?;

        // Create converter
        let convert = gst::ElementFactory::make("audioconvert")
            .build()
            .map_err(|_| {
                RecorderError::ElementCreationError("Cannot create audio converter".to_string())
            })?;

        // Add to pipeline
        pipeline
            .add_many([&removesilence, &valve, &convert])
            .map_err(|_| {
                RecorderError::PipelineError(
                    "Cannot add silence removal elements to pipeline".to_string(),
                )
            })?;

        // Link elements
        previous_element.link(&removesilence)?;
        removesilence.link(&valve)?;
        valve.link(&convert)?;

        debug!("Silence removal plugin added to pipeline");
        // Return the last element in the chain
        Ok(convert)
    }

    fn configure(&mut self, config: HashMap<String, String>) -> PluginResult<()> {
        let mut updated = false;

        if let Some(threshold_str) = config.get("threshold") {
            match threshold_str.parse::<i32>() {
                Ok(threshold) => {
                    if !(-70..=70).contains(&threshold) {
                        let err = PluginError::ParameterError(
                            "Threshold must be between -70 and 70 dB".to_string(),
                        );
                        error!("{}", err);
                        return Err(err);
                    }
                    self.threshold = threshold;
                    debug!("Updated silence removal threshold: {} dB", threshold);
                    updated = true;
                }
                Err(_) => {
                    let err = PluginError::ParameterError("Invalid threshold setting".to_string());
                    error!("{}", err);
                    return Err(err);
                }
            }
        }

        if let Some(minimum_silence_time_str) = config.get("minimum_silence_time") {
            match minimum_silence_time_str.parse::<u64>() {
                Ok(time) => {
                    if time > 10_000_000_000 {
                        let err = PluginError::ParameterError(
                            "Minimum silence time must be less than 10 seconds (10,000,000,000 ns)"
                                .to_string(),
                        );
                        error!("{}", err);
                        return Err(err);
                    }
                    self.minimum_silence_time = time;
                    debug!("Updated minimum silence time: {} ns", time);
                    updated = true;
                }
                Err(_) => {
                    let err = PluginError::ParameterError(
                        "Invalid minimum silence time setting".to_string(),
                    );
                    error!("{}", err);
                    return Err(err);
                }
            }
        }

        if let Some(squash_str) = config.get("squash") {
            match squash_str.parse::<bool>() {
                Ok(squash) => {
                    self.squash = squash;
                    debug!("Updated squash setting: {}", squash);
                    updated = true;
                }
                Err(_) => {
                    let err = PluginError::ParameterError("Invalid squash setting".to_string());
                    error!("{}", err);
                    return Err(err);
                }
            }
        }

        if let Some(hysteresis_str) = config.get("hysteresis") {
            match hysteresis_str.parse::<u64>() {
                Ok(hysteresis) => {
                    if hysteresis < 1 {
                        let err = PluginError::ParameterError(
                            "Hysteresis must be greater than 0".to_string(),
                        );
                        error!("{}", err);
                        return Err(err);
                    }
                    self.hysteresis = hysteresis;
                    debug!("Updated hysteresis: {}", hysteresis);
                    updated = true;
                }
                Err(_) => {
                    let err = PluginError::ParameterError("Invalid hysteresis setting".to_string());
                    error!("{}", err);
                    return Err(err);
                }
            }
        }

        if !updated {
            let err =
                PluginError::ConfigurationError("No valid configuration provided".to_string());
            warn!("{}", err);
            return Err(err);
        }

        Ok(())
    }

    fn handle_message(&self, msg: &gst::Message) -> PluginResult<()> {
        if let Some(structure) = msg.structure() {
            if structure.name() == "removesilence" {
                // Handle silence detection messages
                if let Ok(silence_detected) = structure.get::<u64>("silence_detected") {
                    debug!("Silence detected, PTS: {}", silence_detected);
                }
                if let Ok(silence_finished) = structure.get::<u64>("silence_finished") {
                    debug!("Silence ended, PTS: {}", silence_finished);
                }
            }
        }

        Ok(())
    }
}
