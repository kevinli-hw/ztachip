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

entity barrel_shifter_a is
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

architecture rtl of barrel_shifter_a is
signal distance:unsigned(DIST_WIDTH-1 downto 0);
signal shift_left:std_logic_vector((DATA_WIDTH-1) downto 0);
signal shift_right:std_logic_vector((DATA_WIDTH-1) downto 0);

-- rounding
signal rounding_bit:std_logic;
signal sticky_bit:std_logic;
signal do_round:std_logic;
signal shift_right_rounded:std_logic_vector((DATA_WIDTH-1) downto 0);

begin

distance <= unsigned(distance_in);

-- Rounding bit: data_in[distance-1] (MSB of shifted-out portion)
process(data_in, distance_in)
variable dist : integer;
begin
   rounding_bit <= '0';
   dist := to_integer(unsigned(distance_in));
   if (dist > 0 and rounding_in = '1')then
      rounding_bit <= data_in(dist - 1);
   end if;
end process;

-- Sticky bit: OR of data_in[distance-2 : 0] (remaining shifted-out bits below rounding bit)
process(data_in, distance_in)
variable dist : integer;
variable s : std_logic;
begin
   s := '0';
   dist := to_integer(unsigned(distance_in));
   if rounding_in = '1' then
      for i in 0 to DATA_WIDTH-3 loop
         if i + 1 < dist then
            s := s or data_in(i);
         end if;
      end loop;
   end if;
   sticky_bit <= s;
end process;

-- RoundingDivideByPOT: round away from zero
-- Positive: add 1 when rounding_bit=1
-- Negative: add 1 when rounding_bit=1 AND sticky_bit=1 (not exactly 0.5)
do_round <= rounding_bit and (not data_in(DATA_WIDTH-1) or sticky_bit);

shift_right_rounded <= std_logic_vector(unsigned(shift_right) + (to_unsigned(0,DATA_WIDTH-1) & do_round));

data_out <= shift_right_rounded when (direction_in = '1' and rounding_in = '1') else
            shift_right when (direction_in = '1') else
            shift_left;

-- data_out <= shift_right when (direction_in = '1') else shift_left; 

sra_i : SHIFT_RIGHT_A
   GENERIC MAP (
      DATA_WIDTH=>DATA_WIDTH,
      DIST_WIDTH=>DIST_WIDTH
   )
   PORT MAP (
      data_in=>data_in,
      distance_in=>distance,
      data_out=>shift_right
   );

sla_i : SHIFT_LEFT_A
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
