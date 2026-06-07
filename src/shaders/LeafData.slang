module LeafData;

public struct LeafData {
    public uint first_particle_index;
    public uint particle_count;

    public __init(const uint first_particle_index, const uint particle_count){
        this.particle_count = particle_count;
        this.first_particle_index = first_particle_index;
    }
}