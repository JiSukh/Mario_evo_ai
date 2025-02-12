-- Function to read a byte from NES memory
function read_from_cpu(address)
    return emu.read(address, emu.memType.nesMemory, false)
end

-- Function to read the tile at a specific address
function read_tile(address)
    return emu.read(address, emu.memType.nesMemory, false)
end

local prev_player_x = nil
local timer_elapsed = 0
timeout = 0
death_flag = 0

function get_near_tiles(player_x, player_y, layout)
    local tiles = {}
    local tile_width = 8
    local tile_height = 8

    -- Convert player position to screen tile coordinates
    local screen_x = math.floor(player_x / tile_width)
    local screen_y = math.floor(player_y / tile_height)

    -- Function to calculate the correct tile address in memory
    local function get_tile_address(tile_x, tile_y)
        local screen_num = math.floor(layout / 8)  -- Determine screen (0 or 1)
        local screen_x = tile_x % 8  -- X position within the screen
        local screen_base = screen_num == 0 and 0x0500 or 0x05D0  -- Base address for each screen

        return screen_base + (tile_y * 8) + screen_x  -- Calculate the tile address
    end

    -- Loop over the area around the player to get nearby tiles
    for dy = -3, 3 do
        for dx = -1, 5 do
            local tile_x = screen_x + dx
            local tile_y = screen_y + dy
            local tile_address = get_tile_address(tile_x, tile_y)
            local tile = read_tile(tile_address)

            -- Store tile data with a key based on its coordinates
            tiles[string.format("Block_%d_%d", dx, dy)] = tile
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
    --    player_state = read_from_cpu(0x000E),
    --    player_float_state = read_from_cpu(0x001D),
    --    player_vertical_velocity = emu.read(0x009F, emu.memType.nesMemory, false),
    --    player_lives = read_from_cpu(0x075A),
    --    player_horizontal_speed = emu.read(0x0057, emu.memType.nesMemory, true),
    --    player_vertical_speed = read_from_cpu(0x0433),
   --	    power_up_state = read_from_cpu(0x0756),
       layout = read_from_cpu(0x071A),
     --   layout_index = read_from_cpu(0x072C),
     --   timer = read_from_cpu(0x07F8) * 100 + read_from_cpu(0x07F9) * 10 + read_from_cpu(0x07FA),
     --   level = read_from_cpu(0x0760),
     --   world = read_from_cpu(0x075F),

    --    enemies_y0 = read_from_cpu(0x03B9),
      --  enemies_y1 = read_from_cpu(0x03BA),
      --  enemies_y2 = read_from_cpu(0x03BB),
     --   enemies_y3 = read_from_cpu(0x03BC),
       -- enemies_y4 = read_from_cpu(0x03BD),

      --  enemies_x0 = read_from_cpu(0x03AE),
	  --  enemies_x1 = read_from_cpu(0x03AF),
      --  enemies_x2 = read_from_cpu(0x03B0),
      --  enemies_x3 = read_from_cpu(0x03B1),
      --  enemies_x4 = read_from_cpu(0x03B2),

      --  enemy_states0 = emu.read(0x0058, emu.memType.nesMemory, true),
      --  enemy_states1 = emu.read(0x0059, emu.memType.nesMemory, true),
      --  enemy_states2 = emu.read(0x005A, emu.memType.nesMemory, true),
      --  enemy_states3 = emu.read(0x005B, emu.memType.nesMemory, true),
      --  enemy_states4 = emu.read(0x005C, emu.memType.nesMemory, true),
      --  timeout = timeout,
        death_flag = death_flag
    }
    
    -- Get the tiles near Mario
    local near_tiles = get_near_tiles(data.player_x_screen, data.player_y_screen, data.layout)
    for key, value in pairs(near_tiles) do
    	data[key] = value
	end
	
	local tile_size = 16

    for key, tile in pairs(near_tiles) do
        -- Extract tile_x and tile_y from key
        local tile_x, tile_y = key:match("Block_(%-?%d+)_(%-?%d+)")
        tile_x, tile_y = tonumber(tile_x), tonumber(tile_y)

        if tile_x and tile_y then
            -- Convert tile position back to screen pixels
            local screen_x = tile_x * tile_size
            local screen_y = tile_y * tile_size

            -- Draw rectangle if tile is solid (assuming tile ≠ 0 is a block)
            if tile ~= 0 then
                emu.drawRectangle(screen_x+20, screen_y+20, tile_size, tile_size, 0xFFFFFF, true) -- White filled block
            end
        end
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

    if data.timeout == 1 then
        emu.stop(0)
    end
    if data.death_flag == 1 then
        emu.stop(0)
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
		death_flag = 1
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





-- Function to check if the player's x-coordinate has changed
function check_x_change()
    local player_x = read_from_cpu(0x006D) * 255 + read_from_cpu(0x0086)
    
    if prev_player_x and prev_player_x <= player_x then
        timer_elapsed = timer_elapsed + 1
    else
        timer_elapsed = 0
    end
 
    prev_player_x = player_x
    

    if timer_elapsed >= 400 then
        timeout = 1
    end
end

emu.addEventCallback(check_x_change, emu.eventType.startFrame)

