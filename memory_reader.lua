﻿function table_to_string(t)
    local str = "{"
    for i, v in ipairs(t) do
        str = str .. v .. (i < #t and ", " or "")
    end
    return str .. "}"
end

function read_info()
    local data = {
        player_y_screen = read_from_cpu(0x00CE),
        player_y_level = read_from_cpu(0x00B5) * read_from_cpu(0x00CE),

        player_x_screen = read_from_cpu(0x0086),
        player_x_level = read_from_cpu(0x006D),

        player_state = read_from_cpu(0x000E),
        player_float_state = read_from_cpu(0x001D),
        player_vertical_velocity = emu.read(0x009F, emu.memType.cpu, true),
        player_lives = read_from_cpu(0x075A),

        player_horizontal_speed = read_from_cpu(0x0057),
        player_vertical_speed = read_from_cpu(0x0433),

        block_collision = read_from_cpu(0x0490),
        power_up_state = read_from_cpu(0x0756),
        
                enemies_y = {
            read_from_cpu(0x006E),
            read_from_cpu(0x006F),
            read_from_cpu(0x0070),
            read_from_cpu(0x0071),
            read_from_cpu(0x0072)
        },

        enemies_x = {
            read_from_cpu(0x0087),
            read_from_cpu(0x0088),
            read_from_cpu(0x0089),
            read_from_cpu(0x008A),
            read_from_cpu(0x008B)
        },

        enemy_states = {
            read_from_cpu(0x001E),
            read_from_cpu(0x001F),
            read_from_cpu(0x0020),
            read_from_cpu(0x0021),
            read_from_cpu(0x0022),
            read_from_cpu(0x0023)
        },

        scroll_amount = read_from_cpu(0x0775),
        timer = read_from_cpu(0x07F8)
    }
    
    local output = ""
    for key, value in pairs(data) do
        if type(value) == "table" then
            output = output .. key .. ":" .. table_to_string(value) .. "\n"
        else
            output = output .. key .. ":" .. tostring(value) .. "\n"
        end
    end
    local mmap_file = io.open("C:\\Users\\jsukh\\Documents\\Mario_evo_ai\\data", "w")
    if mmap_file then
        mmap_file:write(output)
        mmap_file:flush()
        mmap_file:close()
    else
        print("Error: Could not open file for writing")
    end
end

function read_from_cpu(address)
    return emu.read(address, emu.memType.cpu, false)
end

emu.addEventCallback(read_info, emu.eventType.endFrame)
