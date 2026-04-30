use std::thread::{self, JoinHandle};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ThreadConfigError {
    #[error("failed to set core affinity for core {0}")]
    CoreAffinity(usize),
    #[error("sched_setscheduler failed: errno {0}")]
    SchedSetScheduler(i32),
    #[error("thread spawn failed: {0}")]
    Spawn(#[from] std::io::Error),
}

/// Configuration for a hot-path pipeline thread.
#[derive(Debug, Clone)]
pub struct ThreadConfig {
    /// Core to pin this thread to.
    pub core_id: usize,
    /// SCHED_FIFO priority (1-99). 0 means default scheduling.
    pub sched_priority: u8,
    /// Thread name shown in `ps`/`htop`.
    pub name: String,
}

impl ThreadConfig {
    pub fn new(core_id: usize, sched_priority: u8, name: impl Into<String>) -> Self {
        Self {
            core_id,
            sched_priority,
            name: name.into(),
        }
    }

    /// Spawn a thread with this config, running `f`.
    pub fn spawn<F, T>(self, f: F) -> Result<JoinHandle<T>, ThreadConfigError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let core_id = self.core_id;
        let sched_priority = self.sched_priority;

        let handle = thread::Builder::new()
            .name(self.name.clone())
            .spawn(move || {
                Self::apply_affinity(core_id);
                if sched_priority > 0 {
                    Self::apply_sched_fifo(sched_priority);
                }
                f()
            })?;

        Ok(handle)
    }

    fn apply_affinity(core_id: usize) {
        // SAFETY: sched_setaffinity is safe to call; we pass valid pid=0 (current thread)
        // and a properly-constructed cpu_set_t.
        #[cfg(target_os = "linux")]
        unsafe {
            let mut cpu_set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(core_id, &mut cpu_set);
            let ret = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpu_set);
            if ret != 0 {
                eprintln!(
                    "sched_setaffinity core {} failed: errno {}",
                    core_id,
                    *libc::__errno_location()
                );
            }
        }
    }

    fn apply_sched_fifo(priority: u8) {
        // SAFETY: sched_setscheduler is safe; pid=0 targets the calling thread.
        // Requires CAP_SYS_NICE or appropriate RLIMIT_RTPRIO.
        #[cfg(target_os = "linux")]
        unsafe {
            let param = libc::sched_param {
                sched_priority: priority as i32,
            };
            let ret = libc::sched_setscheduler(0, libc::SCHED_FIFO, &param);
            if ret != 0 {
                eprintln!(
                    "sched_setscheduler FIFO {} failed: errno {}",
                    priority,
                    *libc::__errno_location()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_thread_runs_closure() {
        let cfg = ThreadConfig::new(0, 0, "test-thread");
        let handle = cfg.spawn(|| 42u32).unwrap();
        assert_eq!(handle.join().unwrap(), 42);
    }

    #[test]
    fn thread_name_is_set() {
        let cfg = ThreadConfig::new(0, 0, "named-thread");
        let handle = cfg
            .spawn(|| thread::current().name().unwrap().to_string())
            .unwrap();
        assert_eq!(handle.join().unwrap(), "named-thread");
    }

    /// Spawning with nonzero priority exercises apply_sched_fifo.
    /// Without CAP_SYS_NICE the syscall fails silently (eprintln only),
    /// but the closure still runs — verifying no panic on the error path.
    #[test]
    fn nonzero_priority_no_panic() {
        let cfg = ThreadConfig::new(0, 1, "priority-test");
        let handle = cfg.spawn(|| 99u32).unwrap();
        assert_eq!(handle.join().unwrap(), 99);
    }

    #[test]
    fn clone_produces_equal_config() {
        let cfg = ThreadConfig::new(2, 50, "clone-test");
        let cloned = cfg.clone();
        assert_eq!(cloned.core_id, 2);
        assert_eq!(cloned.sched_priority, 50);
        assert_eq!(cloned.name, "clone-test");
    }

    #[test]
    fn error_display_core_affinity() {
        let e = ThreadConfigError::CoreAffinity(3);
        assert!(e.to_string().contains('3'));
    }

    #[test]
    fn error_display_sched_set_scheduler() {
        let e = ThreadConfigError::SchedSetScheduler(13);
        assert!(e.to_string().contains("13"));
    }

    #[test]
    fn high_core_id_no_panic() {
        // core_id 999 is within cpu_set_t range (1024) but almost certainly
        // doesn't exist — sched_setaffinity will fail silently via eprintln.
        let cfg = ThreadConfig::new(999, 0, "high-core");
        let handle = cfg.spawn(|| 7u32).unwrap();
        assert_eq!(handle.join().unwrap(), 7);
    }

    #[test]
    fn max_priority_no_panic() {
        let cfg = ThreadConfig::new(0, 99, "max-prio");
        let handle = cfg.spawn(|| 88u32).unwrap();
        assert_eq!(handle.join().unwrap(), 88);
    }

    #[test]
    fn debug_format() {
        let cfg = ThreadConfig::new(2, 50, "dbg-thread");
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("ThreadConfig"));
        assert!(dbg.contains("dbg-thread"));
    }

    #[test]
    fn error_spawn_variant() {
        let e = ThreadConfigError::Spawn(std::io::Error::other("test error"));
        assert!(e.to_string().contains("test error"));
    }

    /// Verify that SCHED_FIFO priority is actually applied.
    /// Requires CAP_SYS_NICE or a non-zero RLIMIT_RTPRIO — run with:
    ///   sudo cargo test -p remoteway-core -- --ignored sched_fifo_priority_applied
    #[test]
    #[ignore = "requires CAP_SYS_NICE / elevated RLIMIT_RTPRIO"]
    #[cfg(target_os = "linux")]
    fn sched_fifo_priority_applied() {
        let cfg = ThreadConfig::new(0, 10, "sched-fifo-test");
        let handle = cfg
            .spawn(|| {
                // Read scheduling policy of this thread from /proc/self/status.
                // sched_getscheduler returns SCHED_FIFO (1) if set correctly.
                // SAFETY: sched_getscheduler(0) is safe — 0 means current thread.
                unsafe { libc::sched_getscheduler(0) }
            })
            .unwrap();
        let policy = handle.join().unwrap();
        assert_eq!(
            policy,
            libc::SCHED_FIFO,
            "thread should have SCHED_FIFO policy"
        );
    }
}
