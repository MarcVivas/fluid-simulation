use crate::traits::GpuTask;



impl GpuTask for ClusterAabbConstructor {
    fn profiling_label() -> &'static str {
        "Cluster aabb constructor"
    }
}

pub struct ClusterAabbConstructor {

}

impl ClusterAabbConstructor {
}
