
module HashMap;

public static const uint EMPTY_KEY_VALUE = 0xFFFFFFFF;

uint hash(uint key, const uint hash_map_capacity){
    key = (key ^ 61) ^ (key >> 16);
    key = key + (key << 3);
    key = key ^ (key >> 4);
    key = key * 0x27d4eb2d;
    key = key ^ (key >> 15);
    return key;
}

public struct KeyValue {
    uint key;
    uint value;
};

public void insert_to_map(const uint key, const uint value, RWStructuredBuffer<KeyValue> map, const uint hash_map_capacity){

    // We assume capacity is a Power of 2.
    // This mask replaces the modulo operator.
    const uint mask = hash_map_capacity - 1;
    uint slot_index = hash(key, hash_map_capacity) & mask;
    while(true){

        // Atomic compare and swap to check what's inside the slot
        uint existing_key;
        InterlockedCompareExchange(map[slot_index].key, EMPTY_KEY_VALUE, key, existing_key);

        // If the slot is empty
        if (existing_key == EMPTY_KEY_VALUE || existing_key == key){
            // Write the value
            map[slot_index].value = value;
            // Exit the loop
            break;
        }

        // The slot is occupied by a different KEY
        // Use linear probing (move to the next slot)
        slot_index = (slot_index + 1) & mask;
    }
}

public uint map_get_value(const uint key, StructuredBuffer<KeyValue> map, const uint hash_map_capacity){
    // We assume capacity is a Power of 2.
    // This mask replaces the modulo operator.
    const uint mask = hash_map_capacity - 1;
    uint slot_index = hash(key, hash_map_capacity) & mask;

    while(true){
        // Get what's inside the slot
        const KeyValue slot = map[slot_index];

        // If the key is the same or empty, the value is returned
        if(slot.key == key || slot.key == EMPTY_KEY_VALUE){
            return slot.value;
        }

        // The slot is occupied by a different valid KEY
        // Use linear probing (move to the next slot)
        slot_index = (slot_index + 1) & mask;
    }
}