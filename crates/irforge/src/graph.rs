use crate::{
    backend::SavedNodes,
    dispatch::{CompilationOptions, KernelGroup, LossType, Node, OptimType, SimpleDType},
    id::{EdgeId, MetaId, NodeId},
    meta::Metadata,
    tensor::{bf16, f16},
};
use briny::raw::{alloc::cast_vec, cast::cast_slice};
use core::{
    cell::RefCell,
    ops::{Index, IndexMut},
};
use fused_gpu::{
    dispatch::{TargetCompilationOptions, backend::Graph as InnerGraph}, errors::{Error, GraphErrorContext},
};
use std::rc::Rc;

#[derive(Debug)]
#[repr(transparent)]
pub struct Graph(pub(crate) Option<InnerGraph>);

impl Index<NodeId> for Graph {
    type Output = Node;

    fn index(&self, index: NodeId) -> &Self::Output {
        &self.inner().nodes[index.0]
    }
}

impl IndexMut<NodeId> for Graph {
    fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
        &mut self.inner_mut().nodes[index.0]
    }
}

impl Graph {
    #[inline]
    fn inner(&self) -> &InnerGraph {
        self.0.as_ref().expect("graph already borrowed")
    }

    #[inline]
    fn inner_mut(&mut self) -> &mut InnerGraph {
        self.0.as_mut().expect("graph already borrowed")
    }

    #[inline]
    #[must_use]
    pub const fn new(loss: LossType, optim: OptimType) -> Self {
        Self(Some(InnerGraph::new(loss, optim)))
    }

    #[inline]
    pub fn set_target(&mut self, target: TargetCompilationOptions) {
        self.inner_mut().target = target;
    }

    #[inline]
    pub fn get_target(&self) -> TargetCompilationOptions {
        self.inner().target
    }

    #[inline]
    #[must_use]
    pub fn get_adjacent(&self, node: NodeId, user: NodeId) -> Vec<NodeId> {
        cast_vec(self.inner().get_adjacent(node.0, user.0))
    }

    #[inline]
    pub fn get_edge(&self, node: NodeId, user: NodeId) -> Option<EdgeId> {
        Some(EdgeId(self.inner().get_edge(node.0, user.0)?))
    }

    #[inline]
    #[must_use]
    pub fn is_edge(&self, node: NodeId, user: NodeId, edge: EdgeId) -> bool {
        self.inner().is_edge(node.0, user.0, edge.0)
    }

    #[inline]
    #[must_use]
    pub fn is_lhs_edge(&self, node: NodeId, user: NodeId) -> bool {
        self.inner().is_lhs_edge(node.0, user.0)
    }

    #[inline]
    #[must_use]
    pub fn is_rhs_edge(&self, node: NodeId, user: NodeId) -> bool {
        self.inner().is_rhs_edge(node.0, user.0)
    }

    #[inline]
    #[must_use]
    pub fn compute_saved_nodes(&self) -> SavedNodes {
        SavedNodes(self.inner().compute_saved_nodes())
    }

    #[inline]
    pub fn topo_sort(&mut self) -> Result<(), Error<GraphErrorContext>> {
        self.inner_mut().topo_sort()
    }

    #[inline]
    pub fn validate(&self, meta: Metadata) -> Result<(), Vec<Error<GraphErrorContext>>> {
        self.inner().validate(meta.0)
    }

    #[inline]
    pub fn lower(
        &mut self,
        meta: Metadata,
        options: &CompilationOptions,
        saved: &SavedNodes,
    ) -> Result<KernelGroup, Error<GraphErrorContext>> {
        self.inner_mut().rebuild_outputs();

        self.inner()
            .lower(meta.0, options, &saved.0)
            .map_err(move |e| e.into())
    }

    pub fn define<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        struct DefineGuard {
            previous: Option<Rc<DefineOps>>,
        }

        impl Drop for DefineGuard {
            fn drop(&mut self) {
                CURRENT_GRAPH.with(|current| {
                    current.replace(self.previous.take());
                });
            }
        }

        let inner = self.0.take().expect("graph already borrowed");

        let graph = Rc::new(RefCell::new(inner));

        let ops = Rc::new(DefineOps {
            graph: graph.clone(),
        });

        let previous = CURRENT_GRAPH.with(|current| current.replace(Some(ops)));

        let res = {
            let _guard = DefineGuard { previous };
            f()
        };

        let graph = Rc::try_unwrap(graph)
            .expect("graph still referenced")
            .into_inner();

        self.0 = Some(graph);

        res
    }
}

thread_local! {
    static CURRENT_GRAPH: RefCell<Option<Rc<DefineOps>>> = const { RefCell::new(None) };
}

pub fn get_graph() -> GraphGuard {
    CURRENT_GRAPH.with(|current| {
        let ops = current
            .borrow()
            .as_ref()
            .expect("no graph is currently being defined")
            .clone();

        GraphGuard { ops }
    })
}

struct DefineOps {
    graph: Rc<RefCell<InnerGraph>>,
}

pub struct GraphGuard {
    ops: Rc<DefineOps>,
}

impl GraphGuard {
    pub fn with<R>(&self, f: impl FnOnce(&InnerGraph) -> R) -> R {
        let graph = self.ops.graph.borrow();
        f(&graph)
    }

    pub fn with_mut<R>(&self, f: impl FnOnce(&mut InnerGraph) -> R) -> R {
        let mut graph = self.ops.graph.borrow_mut();
        f(&mut graph)
    }
}

pub fn input(shape: &[MetaId], dtype: SimpleDType) -> NodeId {
    get_graph().with_mut(|graph| NodeId(graph.input(cast_slice(shape), dtype)))
}

pub fn constant_f16(data: f16) -> NodeId {
    get_graph().with_mut(|graph| NodeId(graph.constant_f16(data)))
}

pub fn constant_bf16(data: bf16) -> NodeId {
    get_graph().with_mut(|graph| NodeId(graph.constant_bf16(data)))
}

pub fn constant_f32(data: f32) -> NodeId {
    get_graph().with_mut(|graph| NodeId(graph.constant_f32(data)))
}

pub fn constant_u32(data: u32) -> NodeId {
    get_graph().with_mut(|graph| NodeId(graph.constant_u32(data)))
}

pub fn constant_i32(data: i32) -> NodeId {
    get_graph().with_mut(|graph| NodeId(graph.constant_i32(data)))
}

macro_rules! impl_op {
    ($op:ident, $($node:ident),*) => {
        pub fn $op($($node: NodeId),*) -> NodeId {
            get_graph().with_mut(|graph| {
                NodeId(irforge_nn::op::$op(graph, $($node.0),*))
            })
        }
    };
}

impl_op!(mul, a, b);
impl_op!(add, a, b);
impl_op!(div, a, b);
impl_op!(sub, a, b);
impl_op!(matmul, a, b);
impl_op!(abs, x);
impl_op!(exp, x);
impl_op!(gelu, x);
impl_op!(relu, x);
impl_op!(elu, x);
impl_op!(log, x);
impl_op!(neg, x);
impl_op!(tanh, x);
impl_op!(softmax, x);
