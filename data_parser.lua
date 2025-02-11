-- Constants
local TILE_START = 0x0500
local TILE_END = 0x069F
local TILE_SIZE = 8
local TILE_MID_Y = 0x05D0
local SCREEN_WIDTH_TILES = 16
local SCREEN_HEIGHT_TILES = 13-- memory is 32x13 (2 stacks of 16x13 memory)
local TILE_COUNT = TILE_END - TILE_START + 1  -- Number of tiles in memory


local mario_x
local mario_y

local last_tiles = {}
local near_tiles = {}


function update_near_tiles(tile_x, tile_y)
    local mario_x = math.floor(tile_x)
    local mario_y = math.floor(tile_y)
    
    local WORLD_WIDTH = 32  -- The world wraps at 32 tiles

    -- Function to compute the shortest wrapped distance
    local function wrapped_distance(x1, x2, wrap_size)
        local direct = math.abs(x1 - x2)
        local wrapped = wrap_size - direct
        return math.min(direct, wrapped)
    end
    
    --remove old keys
    for key, _ in pairs(near_tiles) do
        local tile_check_x, tile_check_y = key:match("([^_]+)_([^_]+)")
        tile_check_x, tile_check_y = tonumber(tile_check_x), tonumber(tile_check_y)


        if tile_check_x < mario_x - 1 or tile_check_x > mario_x + 7 or tile_check_y < mario_y - 4 or tile_check_y > mario_y + 4 then
            near_tiles[key] = nil  -- Remove out of range tiles
        end

    end
    
	--check in area around mario
    for dx = -1, 7, 1 do
        for dy = -7, 4, 1 do
            local tile_check_x = (mario_x + dx) % WORLD_WIDTH  -- Ensure wrapping
            local tile_check_y = 8 + dy  -- Y does not wrap

            -- Ensure tile_check_x is always positive (Lua's % can be negative for negative numbers)
            if tile_check_x < 0 then
                tile_check_x = tile_check_x + WORLD_WIDTH
            end
            
            
            local key = tile_check_x .. "_" .. tile_check_y

			if last_tiles[key] then
                near_tiles[key] = last_tiles[key]
            end

			--draw Area check
			--emu.drawRectangle(tile_check_x*TILE_SIZE, tile_check_y*TILE_SIZE, TILE_SIZE, TILE_SIZE, 0x0000FF, true, 1)
        end
    end
    
end


function update_tiles(address, value)
    local tile_index = 0
    local tile_y = 0
    local tile_x = 0
    local offset = 0

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
    local mario_y_read = emu.read(0x00CE, emu.memType.nesMemory, false) -- Y position
	
	
    -- Calculate Mario's global X position in pixels
    local global_mario_x = (screen_index * 256) + mario_x_read

    -- Convert to tile-based coordinates
    local tile_x = global_mario_x / TILE_SIZE 
    local tile_y = mario_y_read / TILE_SIZE
    

    -- Adjust for the 32x13 tile grid
    tile_x = (tile_x % 64) / 2 -- Wrap around within the 32-tile width
    tile_y = (tile_y)/2
    
    mario_x = tile_x
    mario_y = tile_y
    
	update_near_tiles(tile_x,tile_y)

    -- Draw Mario as a rectangle in the tile system
    emu.drawRectangle(tile_x * TILE_SIZE, tile_y * TILE_SIZE, TILE_SIZE, -TILE_SIZE, 0x00FFFF, false, 1)
end

function draw_near_tiles()
    for k, v in pairs(near_tiles) do
    	if v.value ~= 0 and v.x >= 0 then
        	emu.drawRectangle(v.x*TILE_SIZE, v.y*TILE_SIZE, TILE_SIZE, TILE_SIZE, 0xFFFFFF, false, 1)
        end
    end
end

function draw_enemies()
	local enemies = {}
    for i = 0, 5 do
    	local screen_index = emu.read(0x006E + i, emu.memType.nesMemory, false)
        local enemy_x = emu.read(0x0087 + i, emu.memType.nesMemory, false)
        local enemy_y = emu.read(0x00CF + i, emu.memType.nesMemory, false)
        local global_enemy_x = (screen_index * 256) + enemy_x

	    local tile_x = global_enemy_x / TILE_SIZE 
    	local tile_y = enemy_y / TILE_SIZE

	    tile_x = (tile_x % 64) / 2 -- Wrap around within the 32-tile width
    	tile_y = (tile_y) / 2
		
		
		if tile_x < mario_x + 7 and tile_x > mario_x -1 then
			emu.drawRectangle(tile_x*TILE_SIZE, tile_y*TILE_SIZE-4, TILE_SIZE, -TILE_SIZE, 0xFF0000, false, 1)  -- Red enemy
    	end
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

emu.addEventCallback(draw_mario, emu.eventType.endFrame)
emu.addEventCallback(draw_near_tiles, emu.eventType.endFrame)
emu.addEventCallback(draw_enemies, emu.eventType.endFrame)
