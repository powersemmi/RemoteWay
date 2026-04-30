//! Generated Wayland protocol bindings for ext-image-capture.
//!
//! These are staging protocols not yet available in wayland-protocols crates.
//! Generated from vendored XML files via `wayland-scanner`.

#![allow(
    dead_code,
    non_camel_case_types,
    unused_unsafe,
    unused_variables,
    non_upper_case_globals,
    non_snake_case,
    unused_imports,
    missing_docs,
    clippy::all
)]

/// ext-foreign-toplevel-list-v1 (dependency of ext-image-capture-source).
pub mod ext_foreign_toplevel_list_v1 {
    pub mod client {
        use wayland_client;
        use wayland_client::protocol::*;

        pub mod __interfaces {
            use wayland_client::protocol::__interfaces::*;
            wayland_scanner::generate_interfaces!("protocols/ext-foreign-toplevel-list-v1.xml");
        }
        use self::__interfaces::*;

        wayland_scanner::generate_client_code!("protocols/ext-foreign-toplevel-list-v1.xml");
    }
}

/// ext-image-capture-source-v1.
pub mod ext_image_capture_source_v1 {
    pub mod client {
        use crate::protocols::ext_foreign_toplevel_list_v1::client::*;
        use wayland_client;
        use wayland_client::protocol::*;

        pub mod __interfaces {
            use crate::protocols::ext_foreign_toplevel_list_v1::client::__interfaces::*;
            use wayland_client::protocol::__interfaces::*;
            wayland_scanner::generate_interfaces!("protocols/ext-image-capture-source-v1.xml");
        }
        use self::__interfaces::*;

        wayland_scanner::generate_client_code!("protocols/ext-image-capture-source-v1.xml");
    }
}

/// ext-image-copy-capture-v1.
pub mod ext_image_copy_capture_v1 {
    pub mod client {
        use crate::protocols::ext_image_capture_source_v1::client::*;
        use wayland_client;
        use wayland_client::protocol::*;

        pub mod __interfaces {
            use crate::protocols::ext_image_capture_source_v1::client::__interfaces::*;
            use wayland_client::protocol::__interfaces::*;
            wayland_scanner::generate_interfaces!("protocols/ext-image-copy-capture-v1.xml");
        }
        use self::__interfaces::*;

        wayland_scanner::generate_client_code!("protocols/ext-image-copy-capture-v1.xml");
    }
}
