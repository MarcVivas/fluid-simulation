
module HashMap;

public static const uint EMPTY_KEY = 0xFFFFFFFF;

uint hash(const uint key, const uint hash_map_capacity){
    return key % hash_map_capacity;
}

public struct KeyValue {
    uint key;
    uint value;
};

public void insert_to_map(const uint key, const uint value, RWStructuredBuffer<KeyValue> map, const uint hash_map_capacity){

    uint slot_index = hash(key, hash_map_capacity);

    while(true){

        // Atomic compare and swap to check what's inside the slot
        uint existing_key;
        InterlockedCompareExchange(map[slot_index].key, EMPTY_KEY, key, existing_key);

        // If the slot is empty
        if (existing_key == EMPTY_KEY || existing_key == key){
            // Write the value
            map[slot_index].value = value;
            // Exit the loop
            break;
        }

        // The slot is occupied by a different KEY
        // Use linear probing (move to the next slot)
        slot_index = (slot_index + 1) % hash_map_capacity;
    }
}
