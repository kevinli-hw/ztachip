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

---
-- Float to integer conversion
---

library std;
use std.standard.all;
LIBRARY ieee;
USE ieee.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.ztachip_pkg.all;

entity fp2int is
   generic
   (
      WIDTH:integer;
      LATENCY:natural
   );
   port 
   (
      SIGNAL reset_in    : in std_logic;
      SIGNAL clock_in    : in std_logic;
      SIGNAL x_in        : in fp32_t;
      SIGNAL y_out       : out std_logic_vector(WIDTH-1 downto 0)
   );
end fp2int;

architecture rtl of fp2int is
signal mantissa:std_logic_vector(WIDTH-2 downto 0);
signal exp:unsigned(7 downto 0);
signal sign:std_logic;
signal shift:unsigned(7 downto 0);
signal mantissa_shift:std_logic_vector(WIDTH-2 downto 0);
signal y:std_logic_vector(WIDTH-1 downto 0);

signal mantissa_r:std_logic_vector(WIDTH-2 downto 0);
signal zero_r:std_logic;
signal zero_rr:std_logic;
signal shift_r:unsigned(7 downto 0);
signal mantissa_shift_r:std_logic_vector(WIDTH-2 downto 0);
signal exp_rr:unsigned(7 downto 0);
signal exp_r:unsigned(7 downto 0);
signal y_r:std_logic_vector(WIDTH-1 downto 0);
signal sign_r:std_logic;
signal sign_rr:std_logic;
begin

shift_i: SHIFT_RIGHT_L
   generic map
   (
      DIST_WIDTH=>4,
      DATA_WIDTH=>mantissa'length
   )
   port map 
   (
      data_in=>mantissa_r,
      distance_in=>shift_r(3 downto 0),
      data_out=>mantissa_shift
   );

sign <= x_in(31);

exp <= unsigned(x_in(30 downto 23));

mantissa <= '1' & x_in(22 downto 25-WIDTH);

shift <= to_unsigned(125+WIDTH,exp'length)-unsigned(exp);

y_out <= y_r;

process(exp_rr,sign_rr,mantissa_shift_r,zero_rr)
variable y_v:std_logic_vector(WIDTH-1 downto 0);
begin
   if(zero_rr = '1') then
      y_v := (others=>'0');
   elsif(exp_rr <= to_unsigned(126,exp'length)) then
      y_v := (others=>'0');
   elsif(exp_rr >= to_unsigned(126+WIDTH,exp'length)) then
      y_v(WIDTH-2 downto 0) := (others=>'1'); -- max
      y_v(WIDTH-1) := '0';
   else
      y_v := '0' & mantissa_shift_r;
   end if;
   if(sign_rr='1' and zero_rr='0') then
      y <= std_logic_vector(unsigned(not y_v) + to_unsigned(1,y_v'length));
   else
      y <= y_v;
   end if;
end process;

process(reset_in,clock_in)
begin
   if reset_in = '0' then
      mantissa_r <= (others=>'0');
      zero_r <= '0';
      zero_rr <= '0';
      shift_r <= (others=>'0');
      mantissa_shift_r <= (others=>'0');
      exp_rr <= (others=>'0');
      exp_r <= (others=>'0');
      sign_rr <= '0';
      sign_r <= '0';
      y_r <= (others=>'0');
   else
      if clock_in'event and clock_in='1' then
         mantissa_r <= mantissa;
         shift_r <= shift;
         mantissa_shift_r <= mantissa_shift;
         exp_rr <= exp_r;
         exp_r <= exp;
         sign_rr <= sign_r;
         sign_r <= sign;
         y_r <= y;
         if(unsigned(x_in(x_in'length-2 downto 0))=to_unsigned(0,x_in'length-1)) then
            zero_r <= '1';
         else
            zero_r <= '0';
         end if;
         zero_rr <= zero_r;
      end if;
   end if;
end process;

end rtl;