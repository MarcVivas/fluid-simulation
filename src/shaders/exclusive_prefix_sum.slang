

struct PrefixSumPushConstants{
    uint32_t num_elements;
}

[[vk::push_constant]] PrefixSumPushConstants pc; 

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(){

}