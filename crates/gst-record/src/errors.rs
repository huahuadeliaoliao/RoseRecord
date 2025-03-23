// errors.rs
use thiserror::Error;

/// Recorder related error types
#[derive(Error, Debug)]
pub enum RecorderError {
    #[error("GStreamer error: {0}")]
    GstError(String),

    #[error("Element creation failed: {0}")]
    ElementCreationError(String),

    #[error("Element linking failed: {source}")]
    ElementLinkError {
        #[from]
        source: gstreamer::glib::BoolError,
    },

    #[error("Initialization error: {0}")]
    InitializationError(String),

    #[error("State setting error: {0}")]
    StateError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Pipeline error: {0}")]
    PipelineError(String),

    #[error("Plugin error: {0}")]
    PluginError(String),

    #[error("Bus error: {0}")]
    BusError(String),

    #[error("Timeout error: {0}")]
    TimeoutError(String),

    #[error("IO error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Plugin specific errors
#[derive(Error, Debug)]
pub enum PluginError {
    #[error("Plugin configuration error: {0}")]
    ConfigurationError(String),

    #[error("Plugin initialization error: {0}")]
    InitializationError(String),

    #[error("Plugin processing error: {0}")]
    ProcessingError(String),

    #[error("Parameter error: {0}")]
    ParameterError(String),

    #[error("Unknown plugin error: {0}")]
    Unknown(String),
}

// Define conversion from GSteamer error to RecorderError
impl From<gstreamer::glib::Error> for RecorderError {
    fn from(err: gstreamer::glib::Error) -> Self {
        RecorderError::GstError(err.to_string())
    }
}

// Define conversion from PluginError to RecorderError
impl From<PluginError> for RecorderError {
    fn from(err: PluginError) -> Self {
        RecorderError::PluginError(err.to_string())
    }
}

// Define aliases for standard Result types to simplify usage
pub type Result<T> = std::result::Result<T, RecorderError>;
pub type PluginResult<T> = std::result::Result<T, PluginError>;
