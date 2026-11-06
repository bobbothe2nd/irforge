use fused_gpu::dispatch::{CompilationOptions, backend::{self, Metadata as InnerMeta}};

use briny::traits::Pod;

use crate::backend::MetaBinding;

#[derive(Debug)]
pub struct KernelGroup {
    pub(crate) inner: Box<backend::kernel::KernelGroup>,
    pub(crate) options: CompilationOptions,
}

#[derive(Debug)]
pub struct RawMetaBuf<'a> {
    pub lr: f32,
    pub dims: &'a [u32],
}

impl RawMetaBuf<'_> {
    pub fn to_binding(&self) -> Box<MetaBinding> {
        let mut binding = Vec::new();
        binding.push(self.lr.to_bits());
        binding.extend(self.dims);
        let boxed = binding.into_boxed_slice();

        let ptr = Box::into_raw(boxed) as *mut MetaBinding;
        unsafe { Box::from_raw(ptr) }
    }
}

#[derive(Debug, Default, Hash)]
pub struct MetaBuilder(InnerMeta);

impl MetaBuilder {
    /// Defines an empty metadata instance.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self(InnerMeta::new())
    }

    /// Defines a new metadata field, returning the identifier.
    ///
    /// The identifier will always be 1 more than than the previous field, starting from 0.
    pub const fn new_field(&mut self) -> MetaId {
        MetaId(self.0.new_field())
    }

    /// Checks if metadata is valid.
    ///
    /// Valid metadata is defined as:
    ///
    /// - instantiation must be the same length as compile time definition
    /// - all shape fields must be multiples of 16
    ///
    /// If valid, return `Some`. If any of the above criteria fail, return `None`.
    #[must_use]
    #[inline]
    pub fn build(&self, meta: &MetaBinding) -> Option<Metadata> {
        if self.0.validate_meta(&meta.0) {
            Some(Metadata(self.0))
        } else {
            None
        }
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Hash)]
pub struct Metadata(pub(crate) InnerMeta);

macro_rules! id {
    ($name:ident($inner:ty)) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(pub(crate) $inner);

        unsafe impl Pod for $name {}

        impl $name {
            #[must_use]
            #[inline(always)]
            pub const fn to_bits(self) -> $inner {
                self.0
            }

            #[must_use]
            #[inline(always)]
            pub const fn from_bits(bits: $inner) -> Self {
                Self(bits)
            }
        }
    };
}

id!(EdgeId(usize));
id!(NodeId(backend::NodeId));
id!(ParamId(backend::ParamId));
id!(SharedId(backend::SharedId));
id!(MetaId(backend::MetaId));
id!(ValueId(backend::ValueId));
