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
        use self::__interfaces::{
            EXT_FOREIGN_TOPLEVEL_HANDLE_V1_INTERFACE, EXT_FOREIGN_TOPLEVEL_LIST_V1_INTERFACE,
        };

        wayland_scanner::generate_client_code!("protocols/ext-foreign-toplevel-list-v1.xml");
    }
}

/// ext-image-capture-source-v1.
pub mod ext_image_capture_source_v1 {
    pub mod client {
        use crate::protocols::ext_foreign_toplevel_list_v1::client::ext_foreign_toplevel_handle_v1;
        use wayland_client;
        use wayland_client::protocol::wl_output;

        pub mod __interfaces {
            use crate::protocols::ext_foreign_toplevel_list_v1::client::__interfaces::{
                EXT_FOREIGN_TOPLEVEL_HANDLE_V1_INTERFACE, ext_foreign_toplevel_handle_v1_interface,
            };
            use wayland_client::protocol::__interfaces::{
                WL_OUTPUT_INTERFACE, wl_output_interface,
            };
            wayland_scanner::generate_interfaces!("protocols/ext-image-capture-source-v1.xml");
        }
        use self::__interfaces::{
            EXT_FOREIGN_TOPLEVEL_IMAGE_CAPTURE_SOURCE_MANAGER_V1_INTERFACE,
            EXT_IMAGE_CAPTURE_SOURCE_V1_INTERFACE,
            EXT_OUTPUT_IMAGE_CAPTURE_SOURCE_MANAGER_V1_INTERFACE,
        };

        wayland_scanner::generate_client_code!("protocols/ext-image-capture-source-v1.xml");
    }
}

/// ext-image-copy-capture-v1.
pub mod ext_image_copy_capture_v1 {
    pub mod client {
        use crate::protocols::ext_image_capture_source_v1::client::ext_image_capture_source_v1;
        use wayland_client;
        use wayland_client::protocol::{wl_buffer, wl_output, wl_pointer, wl_shm};

        pub mod __interfaces {
            use crate::protocols::ext_image_capture_source_v1::client::__interfaces::{
                EXT_IMAGE_CAPTURE_SOURCE_V1_INTERFACE, ext_image_capture_source_v1_interface,
            };
            use wayland_client::protocol::__interfaces::{
                WL_BUFFER_INTERFACE, WL_POINTER_INTERFACE, wl_buffer_interface,
                wl_pointer_interface,
            };
            wayland_scanner::generate_interfaces!("protocols/ext-image-copy-capture-v1.xml");
        }
        use self::__interfaces::{
            EXT_IMAGE_COPY_CAPTURE_CURSOR_SESSION_V1_INTERFACE,
            EXT_IMAGE_COPY_CAPTURE_FRAME_V1_INTERFACE, EXT_IMAGE_COPY_CAPTURE_MANAGER_V1_INTERFACE,
            EXT_IMAGE_COPY_CAPTURE_SESSION_V1_INTERFACE,
        };

        wayland_scanner::generate_client_code!("protocols/ext-image-copy-capture-v1.xml");
    }
}
