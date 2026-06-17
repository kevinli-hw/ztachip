------------------------------------------------------------------------------
-- Copyright [2014] [Ztachip Technologies Inc]
--
-- Author: Vuong Nguyen
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
-- http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
------------------------------------------------------------------------------

library std;
use std.standard.all;
LIBRARY ieee;
USE ieee.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.ztachip_pkg.all;

entity barrel_shifter_l is
   generic
   (
      DIST_WIDTH : natural;
      DATA_WIDTH : natural
   );
   port 
   (
      direction_in : in std_logic;
      rounding_in  : in std_logic;
      data_in      : in std_logic_vector((DATA_WIDTH-1) downto 0);
      distance_in  : in std_logic_vector((DIST_WIDTH-1) downto 0);
      data_out     : out std_logic_vector((DATA_WIDTH-1) downto 0)
   );
end entity;

architecture rtl of barrel_shifter_l is
signal distance:unsigned(DIST_WIDTH-1 downto 0);
signal shift_left:std_logic_vector((DATA_WIDTH-1) downto 0);
signal shift_right:std_logic_vector((DATA_WIDTH-1) downto 0);
signal rounding_bit:std_logic;
signal shift_right_rounded:std_logic_vector((DATA_WIDTH-1) downto 0);
begin

distance <= unsigned(distance_in);

process(data_in, distance_in)
variable dist_int : integer;
begin
   rounding_bit <= '0';
   dist_int := to_integer(unsigned(distance_in));
   if (dist_int > 0 and rounding_in = '1')then
      rounding_bit <= data_in(dist_int - 1);
   end if;
end process;

shift_right_rounded <= std_logic_vector(unsigned(shift_right) + (to_unsigned(0,DATA_WIDTH-1) & rounding_bit));

data_out <= shift_right_rounded when (direction_in = '1' and rounding_in = '1') else
            shift_right when (direction_in = '1') else
            shift_left;

--data_out <= shift_right when (direction_in = '1') else shift_left; 

sra_i : SHIFT_RIGHT_L
   GENERIC MAP (
      DATA_WIDTH=>DATA_WIDTH,
      DIST_WIDTH=>DIST_WIDTH
   )
   PORT MAP (
      data_in=>data_in,
      distance_in=>distance,
      data_out=>shift_right
   );

sla_i : SHIFT_LEFT_L
   GENERIC MAP (
      DATA_WIDTH=>DATA_WIDTH,
      DIST_WIDTH=>DIST_WIDTH
   )
   PORT MAP (
      data_in=>data_in,
      distance_in=>distance,
      data_out=>shift_left
   );

end rtl;
