-- Constants
local TILE_START = 0x0500
local TILE_END = 0x069F
local TILE_SIZE = 8
local TILE_MID_Y = 0x05D0
local SCREEN_WIDTH_TILES = 16
local last_tiles = {}
local mario_x
local mario_y

local mario = {}
local enemies = {}
local near_tiles = {}
local fitness = {}


function update_near_tiles(tile_x, tile_y)
    local mario_x = math.floor(tile_x)
    local mario_y = math.floor(tile_y)
    
    local WORLD_WIDTH = 32  -- The world wraps at 32 tiles
    
    --remove old keys
    for key, _ in pairs(near_tiles) do
        local tile_check_x, tile_check_y = key:match("([^_]+)_([^_]+)")
        tile_check_x, tile_check_y = tonumber(tile_check_x), tonumber(tile_check_y)

    end
    
    
    for dx = -1, 7, 1 do
        for dy = -7, 4, 1 do
            -- Convert to a fixed grid (always from 0,0 to x,y)
    
            -- World coordinate check
            local tile_check_x = (mario_x + dx) % WORLD_WIDTH  -- Wrapping X
            local tile_check_y = 8 + dy  -- No wrapping for Y
    
            -- Ensure tile_check_x is always positive
            if tile_check_x < 0 then
                tile_check_x = tile_check_x + WORLD_WIDTH
            end
    
            -- Use grid_x and grid_y as the fixed key (instead of world coordinates)
            local key = dx+1 .. "_" .. dy+7
            local old_key = tile_check_x .. "_" .. tile_check_y
    
            if last_tiles[old_key] then
                near_tiles[key] = {x = dx+1, y = dy+7, value = last_tiles[old_key].value}
                --emu.drawRectangle(near_tiles[key].x * TILE_SIZE, near_tiles[key].y * TILE_SIZE, TILE_SIZE, TILE_SIZE, 0x0000FF, true, 1)
            end
            
            emu.drawRectangle(2*TILE_SIZE, (mario_y-1)*TILE_SIZE,TILE_SIZE,-TILE_SIZE,0xFF00FF, false)

        end
    end
end


function update_tiles(address, value)
    local tile_index = 0
    local tile_y = 0
    local tile_x = 0

    -- x/y offsets, dealing with memory space
    if address > TILE_MID_Y then
        tile_index = address - TILE_START
        tile_x = tile_x + (tile_index % SCREEN_WIDTH_TILES) + 16
        tile_y = math.floor(tile_index / SCREEN_WIDTH_TILES) - 13
    else
        tile_index = address - TILE_START
        tile_x = tile_x + (tile_index % SCREEN_WIDTH_TILES)
        tile_y = math.floor(tile_index / SCREEN_WIDTH_TILES)
    end

    local key = tile_x .. "_" .. tile_y

    if value then
        -- Update or insert the tile data using the address as the key
        last_tiles[key] = {x = tile_x, y = tile_y, value = value}  
    end
    
end

--[[
DRAW FUNCTIONS
--]]

function draw_mario()
    local screen_index = emu.read(0x006D, emu.memType.nesMemory, false) -- 'Screen' number for mario
    local mario_x_read = emu.read(0x0086, emu.memType.nesMemory, false) -- Local X position within the 'screen'
    local mario_y_read = emu.read(0x00CE, emu.memType.nesMemory, false) -- Y position0x009F
	local vy = emu.read(0x009F , emu.memType.nesMemory, true)
	local vx = emu.read(0x0057 , emu.memType.nesMemory, true)
	
    -- Calculate Mario's global X position in pixels
    local global_mario_x = (screen_index * 256) + mario_x_read

    -- Convert to tile-based coordinates
    local tile_x = global_mario_x / TILE_SIZE 
    local tile_y = mario_y_read / TILE_SIZE
    
    
    mario = {x = tile_x, y = tile_y, vx= vx, vy=vy}
    mario_x = tile_x
    mario_y = tile_y
    
    -- Adjust for the 32x13 tile grid
    tile_x = (tile_x % 64) / 2 -- Wrap around within the 32-tile width
    tile_y = (tile_y)/2
    

    
	update_near_tiles(tile_x,tile_y)

    -- Draw Mario as a rectangle in the tile system
    --emu.drawRectangle(tile_x * TILE_SIZE, tile_y * TILE_SIZE, TILE_SIZE, -TILE_SIZE, 0x00FFFF, false, 1)
    emu.drawString(120,10,mario_x)
	emu.drawString(120,20,mario_y)
	emu.drawString(120,30,vy)
	emu.drawString(120,40,vx)
end

function draw_near_tiles()
    for k, v in pairs(near_tiles) do
    	if v.value ~= 0 and v.x >= 0 then
        	emu.drawRectangle(v.x*TILE_SIZE, v.y*TILE_SIZE, TILE_SIZE, TILE_SIZE, 0xFFFFFF, false, 1)
        end
    end
end

function draw_enemies()
	local scale = 32
    local mario_tile_x = mario_x
    local mario_tile_y = mario_y
    
    for i = 0, 4 do
        local screen_index = emu.read(0x006E + i, emu.memType.nesMemory, false)
        local enemy_x = emu.read(0x0087 + i, emu.memType.nesMemory, false)
        local enemy_y = emu.read(0x00CF + i, emu.memType.nesMemory, false)
        local enemy_type = emu.read(0x0016 + i, emu.memType.nesMemory, false)
        local enemy_active = emu.read(0x000F + i, emu.memType.nesMemory, false)
        local velocity_y = emu.read(0x00A0 + i, emu.memType.nesMemory, false)
        local velocity_x = emu.read(0x0058 + i, emu.memType.nesMemory, false)
        local global_enemy_x = (screen_index * 256) + enemy_x

        -- Convert to tile coordinates
        local tile_x = global_enemy_x / TILE_SIZE
        local tile_y = enemy_y / TILE_SIZE
		
		--debug
		if enemy_active == 1 then
			--debug
			
            local norm_x = (tile_x - mario_tile_x) / scale
            local norm_y = (tile_y - mario_tile_y) / scale
            
            
	        emu.drawRectangle(tile_x * TILE_SIZE, tile_y * TILE_SIZE, TILE_SIZE, TILE_SIZE, 0xFF0000, false, 1)  -- Red enemy
	        emu.drawString(10,10+i*30,"ex " .. norm_x)
	        emu.drawString(10,20+i*30,"ey " .. norm_y)
	        emu.drawString(60,15+i*30,"type " .. enemy_type)
	        emu.drawString(10,30+i*30,"active " .. enemy_active)
	        emu.drawString(60,25+i*30,"vy " .. velocity_y)
	        
	        --normalisation

            
    		enemies[i] = {x = norm_x, y = norm_y, t = enemy_type, vx=velocity_x, vy= velocity_y}
    	else
    		enemies[i] = {}
        end
    end
end

function get_fitness()
	local d1 = emu.read(0x07F8, emu.memType.nesMemory, false)
	local d2 = emu.read(0x07F9, emu.memType.nesMemory, false)
	local d3 = emu.read(0x07FA, emu.memType.nesMemory, false)
	local lives = emu.read(0x075A, emu.memType.nesMemory, false)
    local screen_index = emu.read(0x006D, emu.memType.nesMemory, false) -- 'Screen' number for mario
    local mario_x_read = emu.read(0x0086, emu.memType.nesMemory, false) -- Local X position within the 'screen'
    
    local global_mario_x = (screen_index * 256) + mario_x_read

    -- Convert to tile-based coordinates
    local tile_x = global_mario_x / TILE_SIZE 
	
	local timer = (d1 * 100) + (d2 * 10) + d3

	
	fitness = {d = tile_x, t = timer, l = lives}
	
	emu.drawString(230,10,lives)
end


local socket = require("socket.core")

local host = "127.0.0.1" 
local port = 5000      
local udp


function connect_client()
	udp = assert(socket.udp())
	udp:setsockname(host,port)
	udp:settimeout(0)
	emu.removeEventCallback(sockevent, emu.eventType.endFrame)
end


function send_data()

    if not udp then
        print("UDP socket is not initialized. Did you call connect_client()?")
        return
    end
	local message = flatten_tables(mario, near_tiles, enemies, fitness)
	
	if message == "no data" then
		return
	end


	udp:sendto(message, host, port) -- Send message
	
	local response, err = udp:receive() -- Try to receive response
	if response then
		parse_response(response)
	else
	    print("No response received:", err)
	end

end


function parse_response(action_str)
    local values = {}
    local threshold = 0.5
    -- Iterate through the numbers in the response string and convert them to floats
    for value in string.gmatch(action_str, "([-?]?%d*%.?%d+)") do
        table.insert(values, tonumber(value))  -- Convert string to float and insert into the table
    end
	
	if values[4] > threshold then
		emu.setInput({a = true},0 )
	else
		emu.setInput({a = false},0)
	end
	if values[1] > threshold then
		emu.setInput({right = true},0)
	else
		emu.setInput({right = false},0)
	end
	if values[2] > threshold then
		emu.setInput({b = true},0)
	else
		emu.setInput({b = false},0)
	end
	if values[3] > threshold then
		emu.setInput({left = true},0)
	else
		emu.setInput({left = false},0)
	end
	
end

function close_client()
	upd:close()
end


function flatten_score(tbl)
    if tbl == nil then
        return "nil"
    end

    local result = {}
    for k, v in pairs(tbl) do
        local formatted = string.format("%s=%s", tostring(k), tostring(v))
        table.insert(result, formatted)
    end
    return "{" .. table.concat(result, ", ") .. "}"
end

function flatten_mario(mario_o)
    local result = {}
    for k, v in pairs(mario_o) do
        -- Simple key-value formatting for Mario table
        local formatted = string.format("{%s=%s}", tostring(k), tostring(v))
        table.insert(result, formatted)
    end
    return "[" .. table.concat(result, ", ") .. "]"
end

function flatten_near_tiles(near_tiles)
    local result = {}
    for key, tile in pairs(near_tiles) do
        local formatted = string.format(
            "{%s=%s}", 
            tostring(key),
            tostring(tile.value)
        )
        table.insert(result, formatted)
    end
    return "[" .. table.concat(result, ", ") .. "]"
end

function flatten_enemies(...)
    local result = {}
    for _, tbl in ipairs({...}) do
        for k, v in pairs(tbl) do  -- Iterate over tables
            if type(v) == "table" then
                -- Handle nested table
                local formatted = string.format(
                    "{x=%s, y=%s, t=%s, vx=%s, vy=%s}",
                    tostring(v.x or "0"),
                    tostring(v.y or "0"),
                    tostring(v.t or "0"),
                    tostring(v.vx or "0"),
                    tostring(v.vy or "0")
                )
                table.insert(result, formatted)
            else
                -- Handle non-table value
                local formatted = string.format("{%s=%s}", tostring(k), tostring(v))
                table.insert(result, formatted)
            end
        end
    end
    return "[" .. table.concat(result, ", ") .. "]"
end

function flatten_tables(mario, tiles, enemies,fitness)
    local mario_msg = flatten_mario(mario)
    local tile_msg = flatten_near_tiles(tiles)
    local enemies_msg = flatten_enemies(enemies)
    local score_msg = flatten_score(fitness)

    -- Ensure each function returns a valid string longer than 1 character
    if #mario_msg <= 10 or #tile_msg <= 50 or #enemies_msg <= 10  or #score_msg <= 7 then
        return "no data"
    end

    local message = "Mario: " .. mario_msg .. " Tile: " .. tile_msg .. " Enemies: " .. enemies_msg .. " Score: " .. score_msg
    return message
end


function wait_for_title_screen()
	x = emu.read(0x0772,emu.memType.nesMemory, false)
	if x == 0x03 then
		emu.write(0x0770,0x01, emu.memType.nesMemory)
		emu.removeEventCallback(wait_title,emu.eventType.inputPolled)
	end
end

--[[ draw all tileset
function draw_last_tiles()
    for k, v in pairs(last_tiles) do
    	if v.value ~= 0 and v.x >= 0 then
        	emu.drawRectangle(v.x*TILE_SIZE, v.y*TILE_SIZE, TILE_SIZE, TILE_SIZE, 0xFFFFFF, false, 1)
        end
    end
end
--]]
--emu.addEventCallback(draw_last_tiles, emu.eventType.endFrame)

--Read memory when tiles are written into RAM
emu.addMemoryCallback(update_tiles, emu.callbackType.write, TILE_START, TILE_END)

wait_title = emu.addEventCallback(wait_for_title_screen, emu.eventType.inputPolled)
emu.addEventCallback(draw_mario, emu.eventType.endFrame)
emu.addEventCallback(draw_enemies, emu.eventType.endFrame)
emu.addEventCallback(get_fitness, emu.eventType.endFrame)
emu.addEventCallback(draw_near_tiles, emu.eventType.endFrame)
sockevent = emu.addEventCallback(connect_client, emu.eventType.endFrame)

emu.addEventCallback(close_client, emu.eventType.scriptEnded)
emu.addEventCallback(send_data, emu.eventType.inputPolled)


