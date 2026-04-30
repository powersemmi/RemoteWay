use crate::frame_handle::FrameHandle;

/// Common interface for a synchronous pipeline stage.
pub trait PipelineStage: Send + Sync {
    fn process(&self, input: FrameHandle) -> FrameHandle;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PassThrough;
    impl PipelineStage for PassThrough {
        fn process(&self, input: FrameHandle) -> FrameHandle {
            input
        }
    }

    struct IncrementLen;
    impl PipelineStage for IncrementLen {
        fn process(&self, input: FrameHandle) -> FrameHandle {
            FrameHandle {
                len: input.len + 1,
                ..input
            }
        }
    }

    #[test]
    fn passthrough_returns_input_unchanged() {
        let stage = PassThrough;
        let input = FrameHandle {
            pool_index: 3,
            len: 1024,
            timestamp_ns: 42,
        };
        assert_eq!(stage.process(input), input);
    }

    #[test]
    fn transform_modifies_handle() {
        let stage = IncrementLen;
        let input = FrameHandle {
            pool_index: 0,
            len: 100,
            timestamp_ns: 0,
        };
        let output = stage.process(input);
        assert_eq!(output.len, 101);
        assert_eq!(output.pool_index, 0);
    }

    #[test]
    fn pipeline_stage_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PassThrough>();
        assert_send_sync::<IncrementLen>();
    }
}
