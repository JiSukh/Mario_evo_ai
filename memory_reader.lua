-- Function to read a byte from NES memory
function read_from_cpu(address)
    return emu.read(address, emu.memType.nesMemory, false)
end

-- Function to read the tile at a specific address
function read_tile(address)
    return emu.read(address, emu.memType.nesMemory, false)
end

-- Function to get tiles near Mario
function get_near_tiles(player_x, player_y)
    local tiles = {}
    local tile_width = 16
    local tile_height = 16

    -- Convert to screen tile coordinates
    local screen_x = math.floor(player_x / tile_width)
    local screen_y = math.floor(player_y / tile_height)

    -- Loop over area around player
    for dy = -3, 3 do
        for dx = -1, 5 do
            local tile_x = screen_x + dx
            local tile_y = screen_y + dy
            -- Example tile memory calculation (depends on game memory layout)
            local tile_address = 0x0500 + (tile_y * 32) + tile_x
            local tile = read_tile(tile_address)
            -- Use tile coordinates as key, tile data as value
            tiles[string.format("Block_%d_%d", tile_x, tile_y)] = tile
        end
    end

    return tiles
end

function read_info()
    local data = {
        player_y_screen = read_from_cpu(0x00CE),
        player_y_level = read_from_cpu(0x00B5) * read_from_cpu(0x00CE),
        player_x_level = (read_from_cpu(0x006D)*255) + (read_from_cpu(0x0086)),
        player_x_screen = read_from_cpu(0x0086),
        player_state = read_from_cpu(0x000E),
        player_float_state = read_from_cpu(0x001D),
        player_vertical_velocity = emu.read(0x009F, emu.memType.nesMemory, false),
        player_lives = read_from_cpu(0x075A),
        player_horizontal_speed = emu.read(0x0057, emu.memType.nesMemory, true),
        player_vertical_speed = read_from_cpu(0x0433),
        power_up_state = read_from_cpu(0x0756),
        x = read_from_cpu(0x071A),
        layout_index = read_from_cpu(0x072C),
        timer = read_from_cpu(0x07F8) * 100 + read_from_cpu(0x07F9) * 10 + read_from_cpu(0x07FA),
        level = read_from_cpu(0x0760),
        world = read_from_cpu(0x075F),

        enemies_y0 = read_from_cpu(0x03B9),
        enemies_y1 = read_from_cpu(0x03BA),
        enemies_y2 = read_from_cpu(0x03BB),
        enemies_y3 = read_from_cpu(0x03BC),
        enemies_y4 = read_from_cpu(0x03BD),

        enemies_x0 = read_from_cpu(0x03AE),
	    enemies_x1 = read_from_cpu(0x03AF),
        enemies_x2 = read_from_cpu(0x03B0),
        enemies_x3 = read_from_cpu(0x03B1),
        enemies_x4 = read_from_cpu(0x03B2),

        enemy_states0 = emu.read(0x0058, emu.memType.nesMemory, true),
        enemy_states1 = emu.read(0x0059, emu.memType.nesMemory, true),
        enemy_states2 = emu.read(0x005A, emu.memType.nesMemory, true),
        enemy_states3 = emu.read(0x005B, emu.memType.nesMemory, true),
        enemy_states4 = emu.read(0x005C, emu.memType.nesMemory, true)
    }
    
    -- Get the tiles near Mario
    local near_tiles = get_near_tiles(data.player_x_screen, data.player_y_screen)
    for key, value in pairs(near_tiles) do
    	data[key] = value
	end


    
    local output = ""
    for key, value in pairs(data) do
        if type(value) == "table" then
            output = output .. key .. ":" .. table.concat(value, ",") .. "\n"
        else
            output = output .. key .. ":" .. tostring(value) .. "\n"
        end
    end
    
    local mmap_file = io.open("data", "w")
    if mmap_file then
        mmap_file:write(output)
        mmap_file:flush()
        mmap_file:close()
    else
        print("Error: Could not open file for writing")
    end
end

function wait_for_title_screen()
	x = emu.read(0x0772,emu.memType.nesMemory, false)
	if x == 0x03 then
		emu.write(0x0770, 0x01, emu.memType.nesMemory)

		emu.removeEventCallback(0,emu.eventType.inputPolled)
		emu.addEventCallback(check_dead, emu.eventType.startFrame)
	end
end

function check_dead()
	x = emu.read(0x075A,emu.memType.nesMemory, false)
	
	if x < 0x02 then
		emu.stop(0)
	end
end

function read_inputs()
	local mmap_file = io.open("input", "r")

    mmap_file:seek("set", 0)
    local content = mmap_file:read("*all") 


    local inputs = {}
    for value in content:gmatch("([^\n]+)") do
        table.insert(inputs, tonumber(value))
    end

	if inputs[1] == 1 then
		emu.setInput({right = true}, 0)
	else
		emu.setInput({right = false}, 0)
	end
	if inputs[2] == 1 then
		emu.setInput({a = true}, 0)
	else
		emu.setInput({a = false}, 0)
	end
	if inputs[3] == 1 then
		emu.setInput({let = true}, 0)
	else
		emu.setInput({left = false}, 0)
	end
end

--emu.addEventCallback(read_inputs, emu.eventType.inputPolled)
--emu.addEventCallback(wait_for_title_screen, emu.eventType.inputPolled)
emu.addEventCallback(read_info, emu.eventType.startFrame)



