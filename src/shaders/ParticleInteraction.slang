
interface ParticleInteraction {

    // Called once before looping through neighbors
    void pre_interaction();

    // Called for every neighboring particle
    void interact();

    // Called after all neighbors have been processed (to finalize values and write back)
    void post_interaction();
}