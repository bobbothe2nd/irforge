use fused_gpu::dispatch::backend::Metadata as InnerMeta;

use crate::id::MetaId;

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
    /// If valid, return `true`. If any of the above criteria fail, return `false`.
    #[must_use]
    #[inline]
    pub fn build(&self, meta: &[u32]) -> Option<Metadata> {
        if self.0.validate_meta(meta) {
            Some(Metadata(self.0))
        } else {
            None
        }
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Hash)]
pub struct Metadata(pub(crate) InnerMeta);
