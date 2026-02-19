------------------------------------------------------------------------------
-- Copyright [2014] [Ztachip Technologies Inc]
--
-- Author: Vuong Nguyen
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except IN compliance with the License.
-- You may obtain a copy of the License at
--
-- http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to IN writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
------------------------------------------------------------------------------

--------
--
-- Implement FPU (floating point unit)
-- Has the following functions
--   Accept FPU commands from RISCV 
--   Fetch input data from SRAM
--   Feed input to ALU to perform floating point operations on input data
--   Write results back to SRAM
-- FPU is fully pipeline for both memory and computing operations resulting
-- in 1 operation per cycle (including memory overhead)
-- input can also be vectors. Then it performas VECTOR_WIDTH operations per cycle
--
--------

library std;
use std.standard.all;
LIBRARY ieee;
USE ieee.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.ztachip_pkg.all;

ENTITY fpu IS
    PORT (
        SIGNAL clock_in                 : IN STD_LOGIC;
        SIGNAL reset_in                 : IN STD_LOGIC;

        -- Bus interface for configuration        
        SIGNAL bus_waddr_in             : IN register_addr_t;
        SIGNAL bus_raddr_in             : IN register_addr_t;
        SIGNAL bus_write_in             : IN STD_LOGIC;
        SIGNAL bus_read_in              : IN STD_LOGIC;
        SIGNAL bus_writedata_in         : IN STD_LOGIC_VECTOR(host_width_c-1 DOWNTO 0);
        SIGNAL bus_readdata_out         : OUT STD_LOGIC_VECTOR(host_width_c-1 DOWNTO 0);
        SIGNAL bus_readdatavalid_out    : OUT STD_LOGIC;
        SIGNAL bus_writewait_out        : OUT STD_LOGIC;
        SIGNAL bus_readwait_out         : OUT STD_LOGIC;

        -- Bus interface to SRAM
        SIGNAL fpu_rd_addr_out          : OUT STD_LOGIC_VECTOR(sram_depth_c-1 DOWNTO 0);
        SIGNAL fpu_wr_addr_out          : OUT STD_LOGIC_VECTOR(sram_depth_c-1 DOWNTO 0);        
        SIGNAL fpu_write_out            : OUT STD_LOGIC;
        SIGNAL fpu_write_wait_in        : IN STD_LOGIC;
        SIGNAL fpu_read_out             : OUT STD_LOGIC;
        SIGNAL fpu_read_wait_in         : IN STD_LOGIC;
        SIGNAL fpu_writedata_out        : OUT STD_LOGIC_VECTOR(ddr_data_width_c-1 DOWNTO 0);
        SIGNAL fpu_writebe_out          : OUT STD_LOGIC_VECTOR(ddr_data_width_c/8-1 downto 0);
        SIGNAL fpu_readdatavalid_in     : IN STD_LOGIC;
        SIGNAL fpu_readdata_in          : IN STD_LOGIC_VECTOR(ddr_data_width_c-1 DOWNTO 0);

        SIGNAL fpu_busy_vm_out          : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);

        SIGNAL fpu_exe_in               : IN STD_LOGIC;

        SIGNAL fpu_exe_vm_in            : IN STD_LOGIC;

        SIGNAL fpu_exe_out              : OUT STD_LOGIC
    );
END fpu;

ARCHITECTURE behavior OF fpu IS

-- FPU instruction definition

type fpu_instruction_t is
record
    opcode:fpu_opcode_t;
    A_addr:unsigned(sram_depth_c-1 downto 0);
    B_addr:unsigned(sram_depth_c-1 downto 0);
    C:fp32_t;
    C_addr:unsigned(sram_depth_c-1 downto 0);
    C_by_value:std_logic;
    C_pending:std_logic;
    C2:fp32_t;
    C2_addr:unsigned(sram_depth_c-1 downto 0);
    C2_by_value:std_logic;
    C2_pending:std_logic;
    X_addr:unsigned(sram_depth_c-1 downto 0);
    Y_addr:unsigned(sram_depth_c-1 downto 0);
    CNT:unsigned(sram_depth_c-1 downto 0);
    LAST:std_logic;
    FAST:std_logic;
    B_enable:std_logic;
    C_enable:std_logic;
    C2_enable:std_logic;
    X_enable:std_logic;
    Y_enable:std_logic;
    B_by_value:std_logic;
    X_by_value:std_logic;
    Y_by_value:std_logic;
    B:fp32_t;
    X:fp32_t;
    Y:fp32_t;
    A_precision:unsigned(2 downto 0);
    A_floor:std_logic;
    A_abs:std_logic;
    B_precision:unsigned(2 downto 0);
    X_double:std_logic;
    Y_double:std_logic;
    C_double:std_logic;
    C2_double:std_logic;
    X_type:register2_t;
    Y_type:register2_t;
end record;

signal fpu_instruction_r:fpu_instruction_t;

signal fpu_next_instruction_r:fpu_instruction_t;

signal fpu_instruction:fpu_instruction_t;

constant fpu_instruction_len_c:integer:=fpu_instruction_r.opcode'length+
                                        fpu_instruction_r.A_addr'length+
                                        fpu_instruction_r.B_addr'length+
                                        fpu_instruction_r.C'length+
                                        fpu_instruction_r.C_addr'length+
                                        1 + -- fpu_instruction_r.C_by_value'length+
                                        1 + --fpu_instruction_r.C_pending'length+
                                        fpu_instruction_r.C2'length+
                                        fpu_instruction_r.C2_addr'length+
                                        1 + -- fpu_instruction_r.C2_by_value'length+
                                        1 + --fpu_instruction_r.C2_pending'length+
                                        fpu_instruction_r.X_addr'length+
                                        fpu_instruction_r.Y_addr'length+
                                        fpu_instruction_r.CNT'length+
                                        1+ --fpu_instruction_r.LAST'length  
                                        1+ --fpu_instruction_r.FAST'length
                                        1+ --fpu_instruction_r.B_enable'length
                                        1+ --fpu_instruction_r.C_enable'length
                                        1+ --fpu_instruction_r.C2_enable'length
                                        1+ --fpu_instruction_r.X_enable'length
                                        1+ --fpu_instruction_r.Y_enable'length
                                        1+ --fpu_instruction_r.B_by_value'length
                                        1+ --fpu_instruction_r.X_by_value'length
                                        1+ --fpu_instruction_r.Y_by_value'length
                                        fpu_instruction_r.B'length+ --fpu_instruction_r.B'length
                                        fpu_instruction_r.X'length+ --fpu_instruction_r.X'length
                                        fpu_instruction_r.Y'length+ --fpu_instruction_r.Y'length  
                                        fpu_instruction_r.A_precision'length+ --fpu_instruction_r.A_precision'length
                                        1+ --fpu_instruction_r.A_floor
                                        1+ --fpu_instruction_r.A_abs
                                        fpu_instruction_r.B_precision'length+ --fpu_instruction_r.B_precision
                                        1+ --fpu_instruction_r.X_double
                                        1+ --fpu_instruction_r.Y_double
                                        1+ --fpu_instruction_r.C_double
                                        1+ --fpu_instruction_r.C2_double
                                        register2_t'length+ --fpu_instruction_r.X_type
                                        register2_t'length; --fpu_instruction_r.Y_type

subtype fpu_instruction_rec_t is std_logic_vector(fpu_instruction_len_c-1 downto 0);

type fpu_instruction_recs_t is array(natural range <>) of fpu_instruction_rec_t;

-- Unpack fp_instruction from FIFO

function unpack_instruction(q_in:fpu_instruction_rec_t) return fpu_instruction_t is  
variable rec_v:fpu_instruction_t;
variable len_v:integer;
begin
    len_v := 0;
    rec_v.opcode := unsigned(q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.opcode'length));
    len_v := len_v + rec_v.opcode'length;
    rec_v.A_addr := unsigned(q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.A_addr'length));
    len_v := len_v + rec_v.A_addr'length;
    rec_v.B_addr := unsigned(q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.B_addr'length));
    len_v := len_v + rec_v.B_addr'length;
    rec_v.C := q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.C'length);
    len_v := len_v + rec_v.C'length;
    rec_v.C_addr := unsigned(q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.C_addr'length));
    len_v := len_v + rec_v.C_addr'length;
    rec_v.C_by_value := q_in(q_in'length-len_v-1);
    len_v := len_v + 1;
    rec_v.C_pending := q_in(q_in'length-len_v-1);
    len_v := len_v + 1;    
    rec_v.C2 := q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.C2'length);
    len_v := len_v + rec_v.C2'length;
    rec_v.C2_addr := unsigned(q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.C2_addr'length));
    len_v := len_v + rec_v.C2_addr'length;
    rec_v.C2_by_value := q_in(q_in'length-len_v-1);
    len_v := len_v + 1;
    rec_v.C2_pending := q_in(q_in'length-len_v-1);
    len_v := len_v + 1;    
    rec_v.X_addr := unsigned(q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.X_addr'length));
    len_v := len_v + rec_v.X_addr'length;
    rec_v.Y_addr := unsigned(q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.Y_addr'length));
    len_v := len_v + rec_v.Y_addr'length;
    rec_v.CNT := unsigned(q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.CNT'length));
    len_v := len_v + rec_v.CNT'length;
    rec_v.LAST := q_in(q_in'length-len_v-1);
    len_v := len_v + 1; 
    rec_v.FAST := q_in(q_in'length-len_v-1);
    len_v := len_v + 1; 
    rec_v.B_enable := q_in(q_in'length-len_v-1);
    len_v := len_v + 1; 
    rec_v.C_enable := q_in(q_in'length-len_v-1);
    len_v := len_v + 1; 
    rec_v.C2_enable := q_in(q_in'length-len_v-1);
    len_v := len_v + 1; 
    rec_v.X_enable := q_in(q_in'length-len_v-1);
    len_v := len_v + 1; 
    rec_v.Y_enable := q_in(q_in'length-len_v-1);
    len_v := len_v + 1;  
    rec_v.B_by_value := q_in(q_in'length-len_v-1);
    len_v := len_v + 1;
    rec_v.X_by_value := q_in(q_in'length-len_v-1);
    len_v := len_v + 1;  
    rec_v.Y_by_value := q_in(q_in'length-len_v-1);
    len_v := len_v + 1;  
    rec_v.B := q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.B'length);
    len_v := len_v + rec_v.B'length;
    rec_v.X := q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.X'length);
    len_v := len_v + rec_v.X'length;
    rec_v.Y := q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.Y'length);
    len_v := len_v + rec_v.Y'length;
    rec_v.A_precision := unsigned(q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.A_precision'length));
    len_v := len_v + rec_v.A_precision'length;
    rec_v.A_floor := q_in(q_in'length-len_v-1);
    len_v := len_v + 1;
    rec_v.A_abs := q_in(q_in'length-len_v-1);
    len_v := len_v + 1;
    rec_v.B_precision := unsigned(q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.B_precision'length));
    len_v := len_v + rec_v.B_precision'length;
    rec_v.X_double := q_in(q_in'length-len_v-1);
    len_v := len_v + 1;
    rec_v.Y_double := q_in(q_in'length-len_v-1);
    len_v := len_v + 1;
    rec_v.C_double := q_in(q_in'length-len_v-1);
    len_v := len_v + 1; 
    rec_v.C2_double := q_in(q_in'length-len_v-1);
    len_v := len_v + 1;
    rec_v.X_type := unsigned(q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.X_type'length));
    len_v := len_v + rec_v.X_type'length;
    rec_v.Y_type := unsigned(q_in(q_in'length-len_v-1 downto q_in'length-len_v-rec_v.Y_type'length));
    len_v := len_v + rec_v.Y_type'length;
    return rec_v;
end unpack_instruction;

-- Pack instruction to FIFO

function pack_instruction(rec_in:fpu_instruction_t;
                        last_in:std_logic;
                        fast_in:std_logic;
                        opcode_in:fpu_opcode_t;
                        floor_in:std_logic;
                        abs_in:std_logic) 
                        return fpu_instruction_rec_t is  
variable len_v:integer;
variable q_v:fpu_instruction_rec_t;
begin
   len_v := 0;

   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.opcode'length) := std_logic_vector(opcode_in);
   len_v := len_v + rec_in.opcode'length;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.A_addr'length) := std_logic_vector(rec_in.A_addr);
   len_v := len_v + rec_in.A_addr'length;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.B_addr'length) := std_logic_vector(rec_in.B_addr);
   len_v := len_v + rec_in.B_addr'length;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.C'length) := std_logic_vector(rec_in.C);
   len_v := len_v + rec_in.C'length;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.C_addr'length) := std_logic_vector(rec_in.C_addr);
   len_v := len_v + rec_in.C_addr'length;
   q_v(q_v'length-len_v-1) := rec_in.C_by_value;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := rec_in.C_pending;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.C2'length) := std_logic_vector(rec_in.C2);
   len_v := len_v + rec_in.C2'length;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.C2_addr'length) := std_logic_vector(rec_in.C2_addr);
   len_v := len_v + rec_in.C2_addr'length;
   q_v(q_v'length-len_v-1) := rec_in.C2_by_value;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := rec_in.C2_pending;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.X_addr'length) := std_logic_vector(rec_in.X_addr);
   len_v := len_v + rec_in.X_addr'length;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.Y_addr'length) := std_logic_vector(rec_in.Y_addr);
   len_v := len_v + rec_in.Y_addr'length;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.CNT'length) := std_logic_vector(rec_in.CNT);
   len_v := len_v + rec_in.CNT'length;
   q_v(q_v'length-len_v-1) := last_in;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := fast_in;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := rec_in.B_enable;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := rec_in.C_enable;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := rec_in.C2_enable;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := rec_in.X_enable;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := rec_in.Y_enable;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := rec_in.B_by_value;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := rec_in.X_by_value;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := rec_in.Y_by_value;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.B'length) := std_logic_vector(rec_in.B);
   len_v := len_v + rec_in.B'length;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.X'length) := std_logic_vector(rec_in.X);
   len_v := len_v + rec_in.X'length;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.Y'length) := std_logic_vector(rec_in.Y);
   len_v := len_v + rec_in.Y'length;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.A_precision'length) := std_logic_vector(rec_in.A_precision);
   len_v := len_v + rec_in.A_precision'length;
   q_v(q_v'length-len_v-1) := floor_in;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := abs_in;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.B_precision'length) := std_logic_vector(rec_in.B_precision);
   len_v := len_v + rec_in.B_precision'length;
   q_v(q_v'length-len_v-1) := rec_in.X_double;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := rec_in.Y_double;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := rec_in.C_double;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1) := rec_in.C2_double;
   len_v := len_v + 1;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.X_type'length) := std_logic_vector(rec_in.X_type);
   len_v := len_v + rec_in.X_type'length;
   q_v(q_v'length-len_v-1 downto q_v'length-len_v-rec_in.Y_type'length) := std_logic_vector(rec_in.Y_type);
   len_v := len_v + rec_in.Y_type'length;
   return q_v;
end pack_instruction;

----
-- Convert proprietary float format to float32
----

subtype zfp2float_retval_t is std_logic_vector(31 downto 0);
function zfp2float(
        int_in:std_logic_vector(15 downto 0)) 
        return zfp2float_retval_t is
variable float_v:std_logic_vector(31 downto 0);
variable exp_v:std_logic_vector(7 downto 0);
variable mantissa_v:std_logic_vector(fp12_mantissa_width_c-1 downto 0);
begin
   exp_v(fp12_exp_width_c-1 downto 0) := int_in(14 downto 14-fp12_exp_width_c+1);
   exp_v(7 downto fp12_exp_width_c) := (others=>'0');
   mantissa_v := int_in(fp12_mantissa_width_c-1 downto 0);
   if(exp_v=std_logic_vector(to_unsigned(0,exp_v'length))) then
      if(mantissa_v=std_logic_vector(to_unsigned(0,fp12_mantissa_width_c))) then
         float_v := (others=>'0');
      else
         float_v(31) := int_in(15);
         float_v(30 downto 0) := (others=>'0');
         if mantissa_v(11) = '1' then
            float_v(22 downto 12) := mantissa_v(10 downto 0);
            float_v(30 downto 23) := std_logic_vector(to_unsigned(138,8));
         elsif mantissa_v(10) = '1' then
            float_v(22 downto 13) := mantissa_v(9 downto 0);
            float_v(30 downto 23) := std_logic_vector(to_unsigned(137,8));
         elsif mantissa_v(9) = '1' then
            float_v(22 downto 14) := mantissa_v(8 downto 0);
            float_v(30 downto 23) := std_logic_vector(to_unsigned(136,8));
         elsif mantissa_v(8) = '1' then
            float_v(22 downto 15) := mantissa_v(7 downto 0);
            float_v(30 downto 23) := std_logic_vector(to_unsigned(135,8));
         elsif mantissa_v(7) = '1' then
            float_v(22 downto 16) := mantissa_v(6 downto 0);
            float_v(30 downto 23) := std_logic_vector(to_unsigned(134,8));
         elsif mantissa_v(6) = '1' then
            float_v(22 downto 17) := mantissa_v(5 downto 0);
            float_v(30 downto 23) := std_logic_vector(to_unsigned(133,8));
         elsif mantissa_v(5) = '1' then
            float_v(22 downto 18) := mantissa_v(4 downto 0);
            float_v(30 downto 23) := std_logic_vector(to_unsigned(132,8));
         elsif mantissa_v(4) = '1' then
            float_v(22 downto 19) := mantissa_v(3 downto 0);
            float_v(30 downto 23) := std_logic_vector(to_unsigned(131,8));
         elsif mantissa_v(3) = '1' then
            float_v(22 downto 20) := mantissa_v(2 downto 0);
            float_v(30 downto 23) := std_logic_vector(to_unsigned(130,8));
         elsif mantissa_v(2) = '1' then
            float_v(22 downto 21) := mantissa_v(1 downto 0);
            float_v(30 downto 23) := std_logic_vector(to_unsigned(129,8));
         elsif mantissa_v(1) = '1' then
            float_v(22 downto 22) := mantissa_v(0 downto 0);
            float_v(30 downto 23) := std_logic_vector(to_unsigned(128,8));
         elsif mantissa_v(0) = '1' then
            float_v(30 downto 23) := std_logic_vector(to_unsigned(127,8));
         else
            float_v := (others=>'0');
         end if;
      end if;
   else
      float_v(31) := int_in(15); -- Sign bit
      float_v(30 downto 23) := std_logic_vector(unsigned(exp_v)+to_unsigned(126+fp12_mantissa_width_c,8));
      float_v(22 downto 22-fp12_mantissa_width_c+1) := mantissa_v;
      float_v(22-fp12_mantissa_width_c downto 0) := (others=>'0');
   end if;
   return float_v;
end function zfp2float;


constant CMD_FIFO_DEPTH:integer:=8;

SIGNAL cmd_fifo_write:fpu_instruction_recs_t(1 downto 0);
SIGNAL cmd_fifo_we:std_logic_vector(1 downto 0);
SIGNAL cmd_fifo_rd:std_logic_vector(1 downto 0);
SIGNAL cmd_fifo_reads:fpu_instruction_recs_t(1 downto 0);
SIGNAL cmd_fifo_read:fpu_instruction_rec_t;
SIGNAL cmd_fifo_empty:std_logic_vector(1 downto 0);
SIGNAL cmd_fifo_full:std_logic_vector(1 downto 0);

-- Read pending bit mask

constant PARM_MAX:integer:=5;

constant PARM_B:integer:=0; -- Pending read for B

constant PARM_X:integer:=1; -- Pending read for X

constant PARM_Y:integer:=2; -- Pending read for Y

constant PARM_C:integer:=3; -- Pending read for C

constant PARM_C2:integer:=4; -- Pending read for C2

constant CACHE_DEPTH:integer:=4; -- Max prefetch window for parameters

constant MAX_CACHE_LEVEL:integer:=6; -- Should have up to 6 values in cache

SIGNAL wregno:register_t;
SIGNAL wregno2:register2_t;
SIGNAL fpu_set_P:register2_t; -- fpu_set P field
SIGNAL fpu_set_M:register2_t; -- fpu_set M bit
SIGNAL fpu_set_W:register2_t; -- fpu_set W bit
SIGNAL rregno:register_t;
SIGNAL rregno2:register2_t;
SIGNAL rden_r:std_logic;
SIGNAL rresp_r:STD_LOGIC_VECTOR(host_width_c-1 DOWNTO 0);

SIGNAL A_addr:unsigned(sram_depth_c-1 DOWNTO 0);
SIGNAL C:fp32_t;
SIGNAL C2:fp32_t;

-- Signals to indicate pending read responses

SIGNAL pending_write:STD_LOGIC_VECTOR(PARM_MAX-1 downto 0);
SIGNAL pending_wrreq:STD_LOGIC;
SIGNAL pending_rdreq:STD_LOGIC;
SIGNAL pending_read:STD_LOGIC_VECTOR(PARM_MAX-1 downto 0);
SIGNAL pending_empty:STD_LOGIC;

-- Signal for parameter signals

SIGNAL B_wrreq:STD_LOGIC;
SIGNAL B_rdreq:STD_LOGIC;
SIGNAL B_rdflush:STD_LOGIC;
SIGNAL B:STD_LOGIC_VECTOR(ddr_data_width_c-1 downto 0);
SIGNAL B_wused:std_logic_vector(CACHE_DEPTH-1 downto 0);
SIGNAL B_pending_r:unsigned(CACHE_DEPTH-1 downto 0);
SIGNAL B_empty:STD_LOGIC;
SIGNAL B_avail:unsigned(CACHE_DEPTH-1 downto 0);

SIGNAL X_wrreq:STD_LOGIC;
SIGNAL X_rdreq:STD_LOGIC;
SIGNAL X_rdflush:STD_LOGIC;
SIGNAL X:STD_LOGIC_VECTOR(ddr_data_width_c-1 downto 0);
SIGNAL X_wused:std_logic_vector(CACHE_DEPTH-1 downto 0);
SIGNAL X_pending_r:unsigned(CACHE_DEPTH-1 downto 0);
SIGNAL X_empty:STD_LOGIC;
SIGNAL X_avail:unsigned(CACHE_DEPTH-1 downto 0);

SIGNAL Y_wrreq:STD_LOGIC;
SIGNAL Y_rdreq:STD_LOGIC;
SIGNAL Y_rdflush:STD_LOGIC;
SIGNAL Y:STD_LOGIC_VECTOR(ddr_data_width_c-1 downto 0);
SIGNAL Y_wused:std_logic_vector(CACHE_DEPTH-1 downto 0);
SIGNAL Y_pending_r:unsigned(CACHE_DEPTH-1 downto 0);
SIGNAL Y_empty:STD_LOGIC;
SIGNAL Y_avail:unsigned(CACHE_DEPTH-1 downto 0);

SIGNAL C_wrreq:STD_LOGIC;
SIGNAL C2_wrreq:STD_LOGIC;

SIGNAL exe_x:fp32_t;
SIGNAL exe_y:fp32_t;
SIGNAL exe_b:fp32_t;

SIGNAL running:STD_LOGIC;
SIGNAL step:unsigned(sram_depth_c-1 DOWNTO 0);

SIGNAL running_r:STD_LOGIC;
SIGNAL busy_r:STD_LOGIC;
SIGNAL busy_rr:STD_LOGIC;
SIGNAL busy:STD_LOGIC;
SIGNAL step_r:unsigned(sram_depth_c-1 DOWNTO 0);

SIGNAL ready:STD_LOGIC;
SIGNAL exe:STD_LOGIC;

SIGNAL fpu_write:STD_LOGIC;
SIGNAL fpu_wr_addr:unsigned(sram_depth_c-1 DOWNTO 0);
SIGNAL fpu_wr_precision:unsigned(2 downto 0);
SIGNAL fpu_writedata:fp32_t;

SIGNAL writedata_r:std_logic_vector(63 downto 0);
SIGNAL writebe_r:std_logic_vector(7 downto 0);
SIGNAL writedata:std_logic_vector(63 downto 0);
SIGNAL writebe:std_logic_vector(7 downto 0);

SIGNAL sram_read_wait:STD_LOGIC;
SIGNAL sram_rd_addr:STD_LOGIC_VECTOR(sram_depth_c-1 DOWNTO 0);
SIGNAL sram_read:STD_LOGIC;
SIGNAL sram_rd_addr_r:STD_LOGIC_VECTOR(sram_depth_c-1 DOWNTO 0);
SIGNAL sram_read_r:STD_LOGIC;


SIGNAL sram_wr_addr:STD_LOGIC_VECTOR(sram_depth_c-1 DOWNTO 0);        
SIGNAL sram_write:STD_LOGIC;
SIGNAL sram_writedata:STD_LOGIC_VECTOR(ddr_data_width_c-1 DOWNTO 0);
SIGNAL sram_writebe:STD_LOGIC_VECTOR(7 downto 0);

SIGNAL sram_wr_addr_r:STD_LOGIC_VECTOR(sram_depth_c-1 DOWNTO 0);        
SIGNAL sram_write_r:STD_LOGIC;
SIGNAL sram_writedata_r:STD_LOGIC_VECTOR(ddr_data_width_c-1 DOWNTO 0);
SIGNAL sram_writebe_r:STD_LOGIC_VECTOR(7 downto 0);

SIGNAL eof:STD_LOGIC;

SIGNAL fpu_eof:STD_LOGIC;

SIGNAL fpu_last:STD_LOGIC;

SIGNAL fpu_fast:STD_LOGIC;

SIGNAL halt_r:STD_LOGIC:='1';

SIGNAL halt:STD_LOGIC;

SIGNAL fpu_readdatavalid_r:STD_LOGIC;

SIGNAL fpu_readdatavalid_rr:STD_LOGIC;

SIGNAL vm_r:STD_LOGIC;

SIGNAL fpu_vm_r:STD_LOGIC;

SIGNAL fpu_exe_r:STD_LOGIC;

SIGNAL fpu_exe_rr:STD_LOGIC;

SIGNAL page_vm:unsigned(sram_depth_c-1 DOWNTO 0);

signal page_vm0_r:std_logic_vector(sram_depth_c-1 DOWNTO 0);

signal page_vm1_r:std_logic_vector(sram_depth_c-1 DOWNTO 0);

signal write_flush_r:std_logic;

signal fpu_exe_pending_r:std_logic_vector(1 downto 0);

signal fpu_busy_vm_r:std_logic_vector(1 downto 0);

signal full:std_logic;

BEGIN

full <= cmd_fifo_full(0) or cmd_fifo_full(1);

fpu_exe_out <= exe;

busy <= (busy_r or busy_rr or fpu_exe_r or fpu_exe_rr);

fpu_busy_vm_out <= fpu_busy_vm_r;

bus_readdata_out <= rresp_r when rden_r='1' else (others=>'Z');

bus_readdatavalid_out <= rden_r;

bus_writewait_out <= '1' when (bus_write_in='1' and full='1' and wregno=to_unsigned(register_fpu_exe_c,register_t'length)) else '0';

bus_readwait_out <= '0';

eof <= not running; -- Last step 

fpu_wr_addr_out <= sram_wr_addr_r;      

fpu_write_out <= sram_write_r;

fpu_writedata_out <= sram_writedata_r;

fpu_writebe_out <= sram_writebe_r;

fpu_rd_addr_out <= sram_rd_addr_r;

fpu_read_out <= sram_read_r;

pending_rdreq <= fpu_readdatavalid_in;

wregno <= unsigned(bus_waddr_in(register_t'length-1 downto 0));

wregno2 <= unsigned(bus_waddr_in(register2_t'length+register_t'length-1 downto register_t'length));

fpu_set_P <= unsigned(bus_waddr_in(register2_t'length+register_t'length-1 downto register_t'length) and register2_fpu_set_P_MASK);

fpu_set_M <= unsigned(bus_waddr_in(register2_t'length+register_t'length-1 downto register_t'length) and register2_fpu_set_M_MASK);

fpu_set_W <= unsigned(bus_waddr_in(register2_t'length+register_t'length-1 downto register_t'length) and register2_fpu_set_W_MASK);

rregno <= unsigned(bus_raddr_in(register_t'length-1 downto 0));

rregno2 <= unsigned(bus_raddr_in(register2_t'length+register_t'length-1 downto register_t'length));

-- Available cache for each parameters

B_avail <= (unsigned(B_wused) + unsigned(B_pending_r)) when (fpu_instruction_r.B_enable='1' and fpu_instruction_r.B_by_value='0') else (others=>'1');

X_avail <= (unsigned(X_wused) + unsigned(X_pending_r)) when (fpu_instruction_r.X_enable='1' and fpu_instruction_r.X_by_value='0') else (others=>'1');

Y_avail <= (unsigned(Y_wused) + unsigned(Y_pending_r)) when (fpu_instruction_r.Y_enable='1' and fpu_instruction_r.Y_by_value='0') else (others=>'1');

ready <= (not running_r) and X_empty and Y_empty and B_empty and pending_empty and (not write_flush_r);

exe <=  running_r and 
        ((not B_empty) or (not fpu_instruction_r.B_enable) or (fpu_instruction_r.B_by_value)) and 
        ((not X_empty) or (not fpu_instruction_r.X_enable) or (fpu_instruction_r.X_by_value)) and 
        ((not Y_empty) or (not fpu_instruction_r.Y_enable) or (fpu_instruction_r.Y_by_value)) and 
        ((fpu_instruction_r.C_by_value) or (not fpu_instruction_r.C_enable)) and
        ((fpu_instruction_r.C2_by_value) or (not fpu_instruction_r.C2_enable));

sram_read_wait <= '0' when (sram_read_r='0' or fpu_read_wait_in='0') else '1';

fpu_instruction <= unpack_instruction(cmd_fifo_read);

page_vm <= unsigned(bus_writedata_in(sram_depth_c-1 downto 0)) + unsigned(page_vm0_r)
        when vm_r='0' 
        else
        unsigned(bus_writedata_in(sram_depth_c-1 downto 0)) + unsigned(page_vm1_r);

---------
-- FIFO to store FPU instructions issued from RISCV to HART0
---------

cmd_fifo_i0:scfifo
	generic map 
	(
        DATA_WIDTH=>fpu_instruction_rec_t'length,
        FIFO_DEPTH=>CMD_FIFO_DEPTH,
        LOOKAHEAD=>TRUE
	)
	port map 
	(
        clock_in=>clock_in,
        reset_in=>reset_in,
        data_in=>cmd_fifo_write(0),
        write_in=>cmd_fifo_we(0),
        read_in=>cmd_fifo_rd(0),
        q_out=>cmd_fifo_reads(0),
        ravail_out=>open,
        wused_out=>open,
        empty_out=>cmd_fifo_empty(0),
        full_out=>cmd_fifo_full(0),
        almost_full_out=>open
	);

---------
-- FIFO to store FPU instructions issued from RISCV to HART1
---------

cmd_fifo_i1:scfifo
	generic map 
	(
        DATA_WIDTH=>fpu_instruction_rec_t'length,
        FIFO_DEPTH=>CMD_FIFO_DEPTH,
        LOOKAHEAD=>TRUE
	)
	port map 
	(
        clock_in=>clock_in,
        reset_in=>reset_in,
        data_in=>cmd_fifo_write(1),
        write_in=>cmd_fifo_we(1),
        read_in=>cmd_fifo_rd(1),
        q_out=>cmd_fifo_reads(1),
        ravail_out=>open,
        wused_out=>open,
        empty_out=>cmd_fifo_empty(1),
        full_out=>cmd_fifo_full(1),
        almost_full_out=>open
	);

--------------
-- Floating point ALU
--------------

falu_core_i : falu_core
    PORT MAP(
        clock_in => clock_in,
        reset_in => reset_in,
        step_in => step_r,
        opcode_in => fpu_instruction_r.opcode,
        input_ena_in => exe,
        input_eof_in => eof,
        input_last_in => fpu_instruction_r.LAST,
        input_fast_in => fpu_instruction_r.FAST,
        A_addr => fpu_instruction_r.A_addr,
        A_precision => fpu_instruction_r.A_precision,
        A_floor => fpu_instruction_r.A_floor,
        A_abs => fpu_instruction_r.A_abs,
        B_in => exe_b,
        C_in => fpu_instruction_r.C,
        C2_in => fpu_instruction_r.C2,
        X_in => exe_x,
        Y_in => exe_y,
        output_ena_out => fpu_write,
        output_addr_out => fpu_wr_addr,
        output_precision_out => fpu_wr_precision,
        output_out => fpu_writedata,
        output_eof_out => fpu_eof,
        output_last_out => fpu_last,
        output_fast_out => fpu_fast
    );

-- FIFO to keep track the read response going to which buffer

read_pending_i:scfifo
	generic map 
	(
        DATA_WIDTH=>PARM_MAX,
        FIFO_DEPTH=>3, -- Dont need more than 8 since sram latency is less than 8
        LOOKAHEAD=>TRUE
	)
	port map 
	(
        clock_in=>clock_in,
        reset_in=>reset_in,
        data_in=>pending_write,
        write_in=>pending_wrreq,
        read_in=>pending_rdreq,
        q_out=>pending_read,
        ravail_out=>open,
        wused_out=>open,
        empty_out=>pending_empty,
        full_out=>open,
        almost_full_out=>open
	);

-- B parameter FIFO

B_fifo_i:scfifo
	generic map 
	(
        DATA_WIDTH=>ddr_data_width_c,
        FIFO_DEPTH=>CACHE_DEPTH,
        LOOKAHEAD=>TRUE
	)
	port map 
	(
        clock_in=>clock_in,
        reset_in=>reset_in,
        data_in=>fpu_readdata_in,
        write_in=>B_wrreq,
        read_in=>B_rdreq,
        flush_in=>B_rdflush,
        q_out=>B,
        ravail_out=>open,
        wused_out=>B_wused,
        empty_out=>B_empty,
        full_out=>open,
        almost_full_out=>open
	);

-- X parameter FIFO

X_fifo_i:scfifo
	generic map 
	(
        DATA_WIDTH=>ddr_data_width_c,
        FIFO_DEPTH=>CACHE_DEPTH,
        LOOKAHEAD=>TRUE
	)
	port map 
	(
        clock_in=>clock_in,
        reset_in=>reset_in,
        data_in=>fpu_readdata_in,
        write_in=>X_wrreq,
        read_in=>X_rdreq,
        flush_in=>X_rdflush,
        q_out=>X,
        ravail_out=>open,
        wused_out=>X_wused,
        empty_out=>X_empty,
        full_out=>open,
        almost_full_out=>open
	);

-- Y parameter FIFO

Y_fifo_i:scfifo
	generic map 
	(
        DATA_WIDTH=>ddr_data_width_c,
        FIFO_DEPTH=>CACHE_DEPTH,
        LOOKAHEAD=>TRUE
	)
	port map 
	(
        clock_in=>clock_in,
        reset_in=>reset_in,
        data_in=>fpu_readdata_in,
        write_in=>Y_wrreq,
        read_in=>Y_rdreq,
        flush_in=>Y_rdflush,
        q_out=>Y,
        ravail_out=>open,
        wused_out=>Y_wused,
        empty_out=>Y_empty,
        full_out=>open,
        almost_full_out=>open
	);

-- Command FIFO 

process(fpu_next_instruction_r,bus_writedata_in,wregno2,bus_write_in,wregno,full,
        fpu_vm_r,ready,cmd_fifo_empty,halt_r,cmd_fifo_reads)
begin
if(vm_r='0') then
    cmd_fifo_write(0) <= pack_instruction(fpu_next_instruction_r,bus_writedata_in(0),bus_writedata_in(1),
                                        unsigned(wregno2(fpu_opcode_t'length-1 downto 0)), --fpu_opcode_t
                                        wregno2(fpu_opcode_t'length), --floor
                                        wregno2(fpu_opcode_t'length+1) --abs
                                        );

    if(bus_write_in='1' and wregno=to_unsigned(register_fpu_exe_c,register_t'length) and full='0') then
        cmd_fifo_we(0) <= '1';
    else 
        cmd_fifo_we(0) <= '0'; 
    end if;

    cmd_fifo_write(1) <= (others=>'0');
    cmd_fifo_we(1) <= '0';
else
    cmd_fifo_write(0) <= (others=>'0');
    cmd_fifo_we(0) <= '0';
    
    cmd_fifo_write(1) <= pack_instruction(fpu_next_instruction_r,bus_writedata_in(0),bus_writedata_in(1),
                                        unsigned(wregno2(fpu_opcode_t'length-1 downto 0)), --fpu_opcode_t
                                        wregno2(fpu_opcode_t'length), --floor
                                        wregno2(fpu_opcode_t'length+1) --abs
                                        );
    if(bus_write_in='1' and wregno=to_unsigned(register_fpu_exe_c,register_t'length) and full='0') then
        cmd_fifo_we(1) <= '1';
    else
        cmd_fifo_we(1) <= '0';
    end if;
end if;

if(fpu_vm_r='0') then
    if((ready='1') and (cmd_fifo_empty(0)='0') and (halt_r='0')) then
        cmd_fifo_rd(0) <= '1';
    else
        cmd_fifo_rd(0) <= '0';
    end if;
    
    cmd_fifo_rd(1) <= '0';

    cmd_fifo_read <= cmd_fifo_reads(0);
else
    cmd_fifo_rd(0) <= '0';
    if((ready='1') and (cmd_fifo_empty(1)='0') and (halt_r='0')) then
        cmd_fifo_rd(1) <= '1';
    else
        cmd_fifo_rd(1) <= '0';
    end if;
    cmd_fifo_read <= cmd_fifo_reads(1);
end if;

end process;

process(fpu_instruction_r,X,step_r)
variable step_v:unsigned(2 downto 0);
variable x_v:std_logic_vector(15 downto 0);
begin
if(fpu_instruction_r.X_enable='1') then
    if(fpu_instruction_r.X_by_value='0') then
        if(fpu_instruction_r.X_double='0') then
            step_v := (others=>'0');
            step_v(1 downto 0) := step_r(1 downto 0)+unsigned(fpu_instruction_r.X_addr(2 downto 1));
            case step_v(1 downto 0) is
                when "00" =>
                    x_v := X(15 downto 0);
                when "01" =>
                    x_v := X(31 downto 16);
                when "10" =>
                    x_v := X(47 downto 32);
                when others =>
                    x_v := X(63 downto 48);
            end case;
            if(fpu_instruction_r.X_type=register2_fpu_set_W_ZFP16) then
                exe_x <= zfp2float(x_v);
            else
                exe_x <= x_v & "0000000000000000";
            end if;
        else
            step_v := (others=>'0');
            step_v(0 downto 0) := step_r(0 downto 0)+unsigned(fpu_instruction_r.X_addr(2 downto 2));
            case step_v(0 downto 0) is
                when "0" =>
                    exe_x <= X(31 downto 0);
                when others =>
                    exe_x <= X(63 downto 32);
            end case;
        end if;
    else
        exe_x <= fpu_instruction_r.X;
    end if;
else
    exe_x <= (others=>'0');
end if;
end process;

process(fpu_instruction_r,Y,step_r)
variable step_v:unsigned(2 downto 0);
variable y_v:std_logic_vector(15 downto 0);
begin
if(fpu_instruction_r.Y_enable='1') then
    if(fpu_instruction_r.Y_by_value='0') then
        if(fpu_instruction_r.Y_double='0') then
            step_v := (others=>'0');
            step_v(1 downto 0) := step_r(1 downto 0)+unsigned(fpu_instruction_r.Y_addr(2 downto 1));
            case step_v(1 downto 0) is
                when "00" =>
                    y_v := Y(15 downto 0);
                when "01" =>
                    y_v := Y(31 downto 16);
                when "10" =>
                    y_v := Y(47 downto 32);
                when others =>
                    y_v := Y(63 downto 48);
            end case;
            if(fpu_instruction_r.Y_type=register2_fpu_set_W_ZFP16) then
                exe_y <= zfp2float(y_v);
            else
                exe_y <= y_v & "0000000000000000";
            end if;
        else
            step_v := (others=>'0');
            step_v(0 downto 0) := step_r(0 downto 0)+unsigned(fpu_instruction_r.Y_addr(2 downto 2));
            case step_v(0 downto 0) is
                when "0" =>
                    exe_y <= Y(31 downto 0);
                when others =>
                    exe_y <= Y(63 downto 32);
            end case;
        end if;
    else
        exe_y <= fpu_instruction_r.Y;
    end if;
else
    exe_y <= (others=>'0');
end if;
end process;

process(fpu_instruction_r,B,step_r)
variable step_v:unsigned(2 downto 0);
begin
if(fpu_instruction_r.B_enable='1') then
    if(fpu_instruction_r.B_by_value='0') then
        if(fpu_instruction_r.B_precision=to_unsigned(2,fpu_instruction_r.B_precision'length)) then
            -- Input is FP16
            step_v := (others=>'0');
            step_v(1 downto 0) := step_r(1 downto 0)+unsigned(fpu_instruction_r.B_addr(2 downto 1));
            case step_v(1 downto 0) is
                when "00" =>
                    exe_b <= B(15 downto 0) & "0000000000000000";
                when "01" =>
                    exe_b <= B(31 downto 16) & "0000000000000000";
                when "10" =>
                    exe_b <= B(47 downto 32) & "0000000000000000";
                when others =>
                    exe_b <= B(63 downto 48) & "0000000000000000";
            end case;
        elsif(fpu_instruction_r.B_precision=to_unsigned(4,fpu_instruction_r.B_precision'length)) then
            -- Input is FP32
            step_v := (others=>'0');
            step_v(0 downto 0) := step_r(0 downto 0)+unsigned(fpu_instruction_r.B_addr(2 downto 2));
            case step_v(0 downto 0) is
                when "0" =>
                    exe_b <= B(31 downto 0);
                when others =>
                    exe_b <= B(63 downto 32);
            end case;
        else
            -- Input is INT8
            step_v := (others=>'0');
            step_v(2 downto 0) := step_r(2 downto 0)+unsigned(fpu_instruction_r.B_addr(2 downto 0));
            case step_v(2 downto 0) is
                when "000" =>
                    exe_b(7 downto 0) <= B(7 downto 0);
                    exe_b(exe_b'length-1 downto 8) <= (others=>B(7));
                when "001" =>
                    exe_b(7 downto 0) <= B(15 downto 8);
                    exe_b(exe_b'length-1 downto 8) <= (others=>B(15));
                when "010" =>
                    exe_b(7 downto 0) <= B(23 downto 16);
                    exe_b(exe_b'length-1 downto 8) <= (others=>B(23));
                when "011" =>
                    exe_b(7 downto 0) <= B(31 downto 24);
                    exe_b(exe_b'length-1 downto 8) <= (others=>B(31));
                when "100" =>
                    exe_b(7 downto 0) <= B(39 downto 32);
                    exe_b(exe_b'length-1 downto 8) <= (others=>B(39));
                when "101" =>
                    exe_b(7 downto 0) <= B(47 downto 40);
                    exe_b(exe_b'length-1 downto 8) <= (others=>B(47));
                when "110" =>
                    exe_b(7 downto 0) <= B(55 downto 48);
                    exe_b(exe_b'length-1 downto 8) <= (others=>B(55));
                when others =>
                    exe_b(7 downto 0) <= B(63 downto 56);
                    exe_b(exe_b'length-1 downto 8) <= (others=>B(63));
            end case;
        end if;
    else
        exe_b <= fpu_instruction_r.B;
    end if;
else
    exe_b <= (others=>'0');
end if;
end process;

------
-- Issue read requests to retrieve parameter values
-- Determine which parameters should be prefetched first
------

process(
    sram_read_wait,running_r,
    B_avail,fpu_instruction_r,
    X_avail,
    Y_avail
)
begin
    pending_write <= (others=>'0');
    pending_wrreq <= '0';
    if(sram_read_wait='0') then
        sram_read <= '0';
        sram_rd_addr <= (others=>'0');
    else
        sram_read <= sram_read_r;
        sram_rd_addr <= sram_rd_addr_r;
    end if;
    if(running_r='1' and sram_read_wait='0') then
        if(fpu_instruction_r.C_enable='1' and fpu_instruction_r.C_by_value='0' and fpu_instruction_r.C_pending='0') then
            pending_write(PARM_C) <= not sram_read_wait;
            pending_wrreq <= not sram_read_wait;
            sram_read <= '1';
            sram_rd_addr <= std_logic_vector(fpu_instruction_r.C_addr(sram_depth_c-1 DOWNTO 3)) & "000";
        elsif(fpu_instruction_r.C2_enable='1' and fpu_instruction_r.C2_by_value='0' and fpu_instruction_r.C2_pending='0') then
            pending_write(PARM_C2) <= not sram_read_wait;
            pending_wrreq <= not sram_read_wait;
            sram_read <= '1';
            sram_rd_addr <= std_logic_vector(fpu_instruction_r.C2_addr(sram_depth_c-1 DOWNTO 3)) & "000";
        elsif(fpu_instruction_r.B_enable='1' and (B_avail < X_avail) and (B_avail < Y_avail) and fpu_instruction_r.B_by_value='0') then
            if(B_avail < MAX_CACHE_LEVEL) then
                pending_write(PARM_B) <= not sram_read_wait;
                pending_wrreq <= not sram_read_wait;
                sram_read <= '1';
                sram_rd_addr <= std_logic_vector(fpu_instruction_r.B_addr(sram_depth_c-1 DOWNTO 0));
            end if; 
        elsif(fpu_instruction_r.X_enable='1' and X_avail < Y_avail and fpu_instruction_r.X_by_value='0') then
            if(X_avail < MAX_CACHE_LEVEL) then
                pending_write(PARM_X) <= not sram_read_wait;
                pending_wrreq <= not sram_read_wait;
                sram_read <= '1';
                sram_rd_addr <= std_logic_vector(fpu_instruction_r.X_addr(sram_depth_c-1 DOWNTO 0));
            end if;
        elsif(fpu_instruction_r.Y_enable='1' and fpu_instruction_r.Y_by_value='0') then
            if(Y_avail < MAX_CACHE_LEVEL) then
                pending_write(PARM_Y) <= not sram_read_wait;
                pending_wrreq <= not sram_read_wait;
                sram_read <= '1';
                sram_rd_addr <= std_logic_vector(fpu_instruction_r.Y_addr(sram_depth_c-1 DOWNTO 0));
            end if;
        end if;
    end if;
end process;

-------
-- Getting read data back
-------
process(fpu_readdatavalid_in,pending_read,fpu_instruction_r,fpu_readdata_in)
begin
B_wrreq <= '0';
X_wrreq <= '0';
Y_wrreq <= '0';
C_wrreq <= '0';
C2_wrreq <= '0';
C <= (others=>'0');
C2 <= (others=>'0');
if(fpu_readdatavalid_in='1') then
    if(pending_read(PARM_C)='1') then
        -- Result for C parameter is coming back
        C_wrreq <= '1';
        if(fpu_instruction_r.C_double='0') then
            if(fpu_instruction_r.C_addr(2 downto 1)="00") then
                C <= fpu_readdata_in(15 downto 0) & "0000000000000000";
            elsif(fpu_instruction_r.C_addr(2 downto 1)="01") then
                C <= fpu_readdata_in(31 downto 16) & "0000000000000000";
            elsif(fpu_instruction_r.C_addr(2 downto 1)="10") then
                C <= fpu_readdata_in(47 downto 32) & "0000000000000000";
            else
                C <= fpu_readdata_in(63 downto 48) & "0000000000000000";
            end if;
        else
            if(fpu_instruction_r.C_addr(2 downto 2)="0") then
                C <= fpu_readdata_in(31 downto 0);
            else
                C <= fpu_readdata_in(63 downto 32);
            end if;
        end if;
    elsif(pending_read(PARM_C2)='1') then
        -- Result for C2 parameter is coming back
        C2_wrreq <= '1';
        if(fpu_instruction_r.C2_double='0') then
            if(fpu_instruction_r.C2_addr(2 downto 1)="00") then
                C2 <= fpu_readdata_in(15 downto 0) & "0000000000000000";
            elsif(fpu_instruction_r.C2_addr(2 downto 1)="01") then
                C2 <= fpu_readdata_in(31 downto 16) & "0000000000000000";
            elsif(fpu_instruction_r.C2_addr(2 downto 1)="10") then
                C2 <= fpu_readdata_in(47 downto 32) & "0000000000000000";
            else
                C2 <= fpu_readdata_in(63 downto 48) & "0000000000000000";
            end if;
        else
            if(fpu_instruction_r.C2_addr(2 downto 2)="0") then
                C2 <= fpu_readdata_in(31 downto 0);
            else
                C2 <= fpu_readdata_in(63 downto 32);
            end if;
        end if;
    elsif(pending_read(PARM_B)='1') then
        B_wrreq <= '1';
    elsif(pending_read(PARM_X)='1') then
        X_wrreq <= '1';
    elsif(pending_read(PARM_Y)='1') then
        Y_wrreq <= '1';
    end if;
end if;
end process;

-----
-- Execute commands
-----
process(
    exe,
    halt_r,fpu_exe_r,
    running_r,
    step_r,
    fpu_instruction_r,
    B_empty,X_empty,Y_empty
)
variable step_v:unsigned(2 downto 0);
begin
running <= running_r;
halt <= halt_r and (not fpu_exe_r);
step <= step_r;
B_rdreq <= '0';
X_rdreq <= '0';
Y_rdreq <= '0';
X_rdflush <= '0';
Y_rdflush <= '0';
B_rdflush <= '0';
A_addr <= fpu_instruction_r.A_addr;
if(exe='1') then
    if((step_r+1)=fpu_instruction_r.CNT) then
        -- Done...
        running <= '0';
        if(fpu_instruction_r.LAST='1') then
            halt <= '1';
        end if;
    end if;
    step <= step_r+1;
    -- Advance destination address
    if(fpu_instruction_r.opcode=register2_fpu_exe_mac_c or
        fpu_instruction_r.opcode=register2_fpu_exe_reciprocal_c or
        fpu_instruction_r.opcode=register2_fpu_exe_inv_sqrt_c or
        fpu_instruction_r.opcode=register2_fpu_exe_exp_c) then
        A_addr <= fpu_instruction_r.A_addr + resize(fpu_instruction_r.A_precision,fpu_instruction_r.A_addr'length);
    elsif (fpu_instruction_r.opcode=register2_fpu_exe_group_max_c) then
        if((std_logic_vector(step_r) and fpu_instruction_r.C(step_r'length-1 downto 0)) = fpu_instruction_r.C(step_r'length-1 downto 0)) then
            A_addr <= fpu_instruction_r.A_addr + resize(fpu_instruction_r.A_precision,fpu_instruction_r.A_addr'length);
        end if;
    end if;
    if(fpu_instruction_r.X_double='0') then
        step_v := (others=>'0');
        step_v(1 downto 0) := step_r(1 downto 0)+unsigned(fpu_instruction_r.X_addr(2 downto 1));
        if(step_v(1 downto 0)="11") then
            -- Every 4th step, we fetch new X  parameter
            X_rdreq <= fpu_instruction_r.X_enable and (not fpu_instruction_r.X_by_value);
        end if;
    else
        step_v := (others=>'0');
        step_v(0 downto 0) := step_r(0 downto 0)+unsigned(fpu_instruction_r.X_addr(2 downto 2));
        if(step_v(0 downto 0)="1") then
            -- Every 4th step, we fetch new X  parameter
            X_rdreq <= fpu_instruction_r.X_enable and (not fpu_instruction_r.X_by_value);
        end if;
    end if;
    if(fpu_instruction_r.Y_double='0') then
        step_v := (others=>'0');
        step_v(1 downto 0) := step_r(1 downto 0)+unsigned(fpu_instruction_r.Y_addr(2 downto 1));
        if(step_v(1 downto 0)="11") then
            -- Every 4th step, we fetch new Y parameter
            Y_rdreq <= fpu_instruction_r.Y_enable and (not fpu_instruction_r.Y_by_value);
        end if;
    else
        step_v := (others=>'0');
        step_v(0 downto 0) := step_r(0 downto 0)+unsigned(fpu_instruction_r.Y_addr(2 downto 2));
        if(step_v(0 downto 0)="1") then
            -- Every 4th step, we fetch new Y parameter
            Y_rdreq <= fpu_instruction_r.Y_enable and (not fpu_instruction_r.Y_by_value);
        end if;
    end if;
    if(fpu_instruction_r.B_precision=to_unsigned(1,fpu_instruction_r.B_precision'length)) then
        -- This is case of INT8
        step_v := (others=>'0');
        step_v(2 downto 0) := step_r(2 downto 0)+unsigned(fpu_instruction_r.B_addr(2 downto 0));
        if(step_v(2 downto 0)="111") then
            -- Every 2th step, we fetch new B parameter
            B_rdreq <= fpu_instruction_r.B_enable and (not fpu_instruction_r.B_by_value);
        end if;
    elsif(fpu_instruction_r.B_precision=to_unsigned(2,fpu_instruction_r.B_precision'length)) then
        -- This is case of FP16
        step_v := (others=>'0');
        step_v(1 downto 0) := step_r(1 downto 0)+unsigned(fpu_instruction_r.B_addr(2 downto 1));
        if(step_v(1 downto 0)="11") then
            -- Every 2th step, we fetch new B parameter
            B_rdreq <= fpu_instruction_r.B_enable and (not fpu_instruction_r.B_by_value);
        end if;
    else
        -- This is case of FP32
        step_v := (others=>'0');
        step_v(0 downto 0) := step_r(0 downto 0)+unsigned(fpu_instruction_r.B_addr(2 downto 2));
        if(step_v(0 downto 0)="1") then
            -- Every 2th step, we fetch new B parameter
            B_rdreq <= fpu_instruction_r.B_enable and (not fpu_instruction_r.B_by_value);
        end if;
    end if;
elsif(running_r='0') then
    -- FPU operation is done. Let empty the FIFO
    X_rdreq <= (not X_empty);
    Y_rdreq <= (not Y_empty);
    B_rdreq <= (not B_empty);
    X_rdflush <= '1';
    Y_rdflush <= '1';
    B_rdflush <= '1';
else
    X_rdreq <= '0';
    Y_rdreq <= '0';
    B_rdreq <= '0';
end if;
end process;

---
-- Write result to SRAM
---

process(fpu_write,fpu_eof,writedata_r,fpu_writedata,fpu_wr_addr,writedata,fpu_wr_precision)
variable complete_v:std_logic;
variable fp16_v:std_logic_vector(15 downto 0);
variable fp16_max_v:std_logic_vector(15 downto 0);
begin
    writedata <= writedata_r;
    writebe <= writebe_r;
    if(fpu_write='1') then
        complete_v := '0';
        if(fpu_wr_precision=4) then
            -- precision is FP32
            if(fpu_wr_addr(2)='0') then
                writedata(31 downto 0) <= fpu_writedata;
                writebe(3 downto 0) <= (others=>'1');
            else
                writedata(63 downto 32) <= fpu_writedata;
                writebe(7 downto 4) <= (others=>'1');
                complete_v := '1';
            end if;
        elsif(fpu_wr_precision=2) then
            -- precision is FP16
            fp16_v := fpu_writedata(31 downto 16);
            fp16_max_v := (others=>'1');
            if(fpu_writedata(15)='1' and fp16_v /= fp16_max_v) then
                fp16_v := std_logic_vector(unsigned(fp16_v) + to_unsigned(1,fp16_v'length));
            end if;
            case fpu_wr_addr(2 downto 1) is
                when "00" => 
                    writedata(15 downto 0) <= fp16_v;  
                    writebe(1 downto 0) <= (others=>'1');
                when "01" => 
                    writedata(31 downto 16) <= fp16_v;  
                    writebe(3 downto 2) <= (others=>'1');
                when "10" => 
                    writedata(47 downto 32) <= fp16_v; 
                    writebe(5 downto 4) <= (others=>'1'); 
                when others => 
                    writedata(63 downto 48) <= fp16_v;
                    writebe(7 downto 6) <= (others=>'1'); 
                    complete_v := '1'; 
            end case;
        else
            -- precision is INT8
            case fpu_wr_addr(2 downto 0) is
                when "000" => 
                    writedata(7 downto 0) <= fpu_writedata(7 downto 0);  
                    writebe(0) <= '1';
                when "001" => 
                    writedata(15 downto 8) <= fpu_writedata(7 downto 0); 
                    writebe(1) <= '1'; 
                when "010" => 
                    writedata(23 downto 16) <= fpu_writedata(7 downto 0);  
                    writebe(2) <= '1';
                when "011" => 
                    writedata(31 downto 24) <= fpu_writedata(7 downto 0); 
                    writebe(3) <= '1';
                when "100" => 
                    writedata(39 downto 32) <= fpu_writedata(7 downto 0); 
                    writebe(4) <= '1';
                when "101" => 
                    writedata(47 downto 40) <= fpu_writedata(7 downto 0); 
                    writebe(5) <= '1';
                when "110" => 
                    writedata(55 downto 48) <= fpu_writedata(7 downto 0); 
                    writebe(6) <= '1';
                when others => 
                    writedata(63 downto 56) <= fpu_writedata(7 downto 0); 
                    writebe(7) <= '1';
                    complete_v := '1'; 
            end case;
        end if;
        if(fpu_eof='1' or complete_v='1') then
            -- Last ALU response, flush it
            if(writebe=std_logic_vector(to_unsigned(0,writebe'length))) then
                sram_write <= '0';
            else
                sram_write <= '1';
            end if;
            sram_writedata <= writedata;
            sram_writebe <= writebe;
            sram_wr_addr <= std_logic_vector(fpu_wr_addr(fpu_wr_addr'length-1 downto 3)) & "000";
        else
            -- Save this write request to combine with next write requests
            sram_write <= '0';
            sram_writedata <= (others=>'0');
            sram_writebe <= (others=>'0');
            sram_wr_addr <= (others=>'0');
        end if;
    else
        -- No write to do this time
        sram_write <= '0';
        sram_writedata <= (others=>'0');
        sram_writebe <= (others=>'0');
        sram_wr_addr <= (others=>'0');
    end if;
end process;

process(clock_in,reset_in)
variable busy_v:std_logic;
variable fpu_exe_v:std_logic;
variable fpu_vm_v:std_logic;
variable fpu_exe_pending_v:std_logic_vector(1 downto 0);
begin
    if reset_in = '0' then
        halt_r <= '1';
        rden_r <= '0';
        rresp_r <= (others=>'0');

        fpu_instruction_r.opcode <= (others=>'0');    
        fpu_instruction_r.A_addr <= (others=>'0');
        fpu_instruction_r.B_addr <= (others=>'0');
        fpu_instruction_r.C <= (others=>'0');
        fpu_instruction_r.C_by_value <= '0';
        fpu_instruction_r.C_addr <= (others=>'0');
        fpu_instruction_r.C2 <= (others=>'0');
        fpu_instruction_r.C2_by_value <= '0';
        fpu_instruction_r.C2_addr <= (others=>'0');
        fpu_instruction_r.X_addr <= (others=>'0');
        fpu_instruction_r.Y_addr <= (others=>'0');
        fpu_instruction_r.CNT <= (others=>'0');
        fpu_instruction_r.C_pending <= '0';
        fpu_instruction_r.C2_pending <= '0';
        fpu_instruction_r.LAST <= '0';
        fpu_instruction_r.FAST <= '0';
        fpu_instruction_r.B_enable <= '0';
        fpu_instruction_r.C_enable <= '0';
        fpu_instruction_r.C2_enable <= '0';
        fpu_instruction_r.X_enable <= '0';
        fpu_instruction_r.Y_enable <= '0';
        fpu_instruction_r.B_by_value <= '0';
        fpu_instruction_r.X_by_value <= '0';
        fpu_instruction_r.Y_by_value <= '0';
        fpu_instruction_r.B <= (others=>'0');
        fpu_instruction_r.X <= (others=>'0');
        fpu_instruction_r.Y <= (others=>'0');
        fpu_instruction_r.A_precision <= (others=>'0');
        fpu_instruction_r.A_floor <= '0';
        fpu_instruction_r.A_abs <= '0';
        fpu_instruction_r.B_precision <= (others=>'0');
        fpu_instruction_r.X_double <= '0';
        fpu_instruction_r.Y_double <= '0';
        fpu_instruction_r.C_double <= '0';
        fpu_instruction_r.C2_double <= '0';
        fpu_instruction_r.X_type <= (others=>'0');
        fpu_instruction_r.Y_type <= (others=>'0');

        fpu_next_instruction_r.opcode <= (others=>'0');  
        fpu_next_instruction_r.A_addr <= (others=>'0');
        fpu_next_instruction_r.B_addr <= (others=>'0');
        fpu_next_instruction_r.C <= (others=>'0');
        fpu_next_instruction_r.C_by_value <= '0';
        fpu_next_instruction_r.C_addr <= (others=>'0');
        fpu_next_instruction_r.C2 <= (others=>'0');
        fpu_next_instruction_r.C2_by_value <= '0';
        fpu_next_instruction_r.C2_addr <= (others=>'0');
        fpu_next_instruction_r.X_addr <= (others=>'0');
        fpu_next_instruction_r.Y_addr <= (others=>'0');
        fpu_next_instruction_r.CNT <= (others=>'0');
        fpu_next_instruction_r.C_pending <= '0';
        fpu_next_instruction_r.C2_pending <= '0';
        fpu_next_instruction_r.LAST <= '0';
        fpu_next_instruction_r.FAST <= '0';
        fpu_next_instruction_r.B_enable <= '0';
        fpu_next_instruction_r.C_enable <= '0';
        fpu_next_instruction_r.C2_enable <= '0';
        fpu_next_instruction_r.X_enable <= '0';
        fpu_next_instruction_r.Y_enable <= '0';
        fpu_next_instruction_r.B_by_value <= '0';
        fpu_next_instruction_r.X_by_value <= '0';
        fpu_next_instruction_r.Y_by_value <= '0';
        fpu_next_instruction_r.B <= (others=>'0');
        fpu_next_instruction_r.X <= (others=>'0');  
        fpu_next_instruction_r.Y <= (others=>'0'); 
        fpu_next_instruction_r.A_precision <= (others=>'0');
        fpu_next_instruction_r.A_floor <= '0';
        fpu_next_instruction_r.A_abs <= '0';
        fpu_next_instruction_r.B_precision <= (others=>'0'); 
        fpu_next_instruction_r.X_double <= '0';
        fpu_next_instruction_r.Y_double <= '0';
        fpu_next_instruction_r.C_double <= '0';
        fpu_next_instruction_r.C2_double <= '0';
        fpu_next_instruction_r.X_type <= (others=>'0');
        fpu_next_instruction_r.Y_type <= (others=>'0');

        B_pending_r <= (others=>'0');
        X_pending_r <= (others=>'0');
        Y_pending_r <= (others=>'0');
        running_r <= '0';
        busy_r <= '0';
        busy_rr <= '0';
        step_r <= (others=>'0');
        writedata_r <= (others=>'0');
        writebe_r <= (others=>'0');
        sram_rd_addr_r <= (others=>'0');
        sram_read_r <= '0';
        sram_wr_addr_r <= (others=>'0');      
        sram_write_r <= '0';
        sram_writedata_r <= (others=>'0');
        sram_writebe_r <= (others=>'0');
        fpu_readdatavalid_r <= '0';
        fpu_readdatavalid_rr <= '0';
        vm_r <= '0';
        fpu_vm_r <= '0';
        fpu_exe_r <= '0';
        fpu_exe_rr <= '0';
        page_vm0_r <= (others=>'0');
        page_vm1_r <= (others=>'0');
        write_flush_r <= '0';
        fpu_exe_pending_r <= (others=>'0');
        fpu_busy_vm_r <= (others=>'0');
    else
        if clock_in'event and clock_in='1' then

            busy_v := busy_r;
            fpu_exe_v := fpu_exe_r;
            fpu_vm_v := fpu_vm_r;
            fpu_exe_pending_v := fpu_exe_pending_r;

            fpu_readdatavalid_r <= fpu_readdatavalid_in;
            fpu_readdatavalid_rr <= fpu_readdatavalid_r;
            halt_r <= halt;
            running_r <= running;
            step_r <= step;
            sram_rd_addr_r <= sram_rd_addr;
            sram_read_r <= sram_read;
            sram_wr_addr_r <= sram_wr_addr;      
            sram_write_r <= sram_write;
            sram_writedata_r <= sram_writedata;
            sram_writebe_r <= sram_writebe;

            -- Latch in write request so to combine with next write
            -- requests
            if(sram_write='1') then
                writedata_r <= (others=>'0');
                writebe_r <= (others=>'0');
            else
                writedata_r <= writedata;
                writebe_r <= writebe;
            end if;
            if(running_r='1') then
                fpu_instruction_r.A_addr <= A_addr;
            end if;
            if(pending_write(PARM_C)='1') then
                fpu_instruction_r.C_pending <= '1';
                fpu_instruction_r.C_by_value <= '0';
            elsif(C_wrreq='1') then
                fpu_instruction_r.C_pending <= '0';
                fpu_instruction_r.C_by_value <= '1';
                fpu_instruction_r.C <= C;
            end if;
            if(pending_write(PARM_C2)='1') then
                fpu_instruction_r.C2_pending <= '1';
                fpu_instruction_r.C2_by_value <= '0';
            elsif(C2_wrreq='1') then
                fpu_instruction_r.C2_pending <= '0';
                fpu_instruction_r.C2_by_value <= '1';
                fpu_instruction_r.C2 <= C2;
            end if;
            if(B_wrreq = '1' and pending_write(PARM_B)='0') then
                B_pending_r <= B_pending_r-1;
            elsif(pending_write(PARM_B)='1' and B_wrreq='0') then
                B_pending_r <= B_pending_r+1;
            end if;
            if(X_wrreq = '1' and pending_write(PARM_X)='0') then
                X_pending_r <= X_pending_r-1;
            elsif(pending_write(PARM_X)='1' and X_wrreq='0') then
                X_pending_r <= X_pending_r+1;
            end if;
            if(Y_wrreq = '1' and pending_write(PARM_Y)='0') then
                Y_pending_r <= Y_pending_r-1;
            elsif(pending_write(PARM_Y)='1' and Y_wrreq='0') then
                Y_pending_r <= Y_pending_r+1;
            end if;

            if(pending_write(PARM_X)='1') then
                fpu_instruction_r.X_addr <= fpu_instruction_r.X_addr+ddr_data_width_c/8;
            end if;
            if(pending_write(PARM_Y)='1') then
                fpu_instruction_r.Y_addr <= fpu_instruction_r.Y_addr+ddr_data_width_c/8;
            end if;
            if(pending_write(PARM_B)='1') then
                fpu_instruction_r.B_addr <= fpu_instruction_r.B_addr+ddr_data_width_c/8;
            end if;

            -----
            -- Process write commands
            ----

            if(fpu_exe_in='1') then
                if(fpu_exe_vm_in='0') then
                  fpu_exe_pending_v(0) := '1'; 
                else
                  fpu_exe_pending_v(1) := '1'; 
                end if;
            end if;
            if(busy='0') then
                if(fpu_vm_r='0') then
                    if(fpu_exe_pending_r(1)='1') then
                        fpu_exe_v := '1';
                        fpu_vm_v := '1';
                        fpu_exe_pending_v(1) := '0'; 
                    elsif(fpu_exe_pending_r(0)='1') then
                        fpu_exe_v := '1';
                        fpu_vm_v := '0';
                        fpu_exe_pending_v(0) := '0'; 
                    else
                        fpu_exe_v := '0';
                    end if;
                else
                    if(fpu_exe_pending_r(0)='1') then
                        fpu_exe_v := '1';
                        fpu_vm_v := '0';
                        fpu_exe_pending_v(0) := '0'; 
                    elsif(fpu_exe_pending_r(1)='1') then
                        fpu_exe_v := '1';
                        fpu_vm_v := '1';
                        fpu_exe_pending_v(1) := '0'; 
                    else
                        fpu_exe_v := '0';
                    end if;
                end if;
            else
                fpu_exe_v := '0';     
            end if;
            if(bus_write_in='1') then
                if wregno=register_vm_toggle_c then
                    vm_r <= not vm_r;
                end if;
                if wregno=register_fpu_set_mem then
                    if(vm_r='0') then
                        page_vm0_r <= bus_writedata_in(sram_depth_c-1 DOWNTO 0);
                    else
                        page_vm1_r <= bus_writedata_in(sram_depth_c-1 DOWNTO 0);
                    end if;
                end if;
                -- Process write commands
                if(wregno=to_unsigned(register_fpu_set_c,register_t'length)) then
                    if(fpu_set_P=register2_fpu_set_P_A) then
                        fpu_next_instruction_r.A_addr <= page_vm;
                        if(fpu_set_W=register2_fpu_set_W_FP32) then
                            fpu_next_instruction_r.A_precision <= to_unsigned(4,fpu_next_instruction_r.A_precision'length); --FP32
                        elsif(fpu_set_W=register2_fpu_set_W_FP16 or fpu_set_W=register2_fpu_set_W_ZFP16) then
                            fpu_next_instruction_r.A_precision <= to_unsigned(2,fpu_next_instruction_r.A_precision'length); --FP16
                        elsif(fpu_set_W=register2_fpu_set_W_INT8) then
                            fpu_next_instruction_r.A_precision <= to_unsigned(1,fpu_next_instruction_r.A_precision'length); -- INT8
                        else
                            fpu_next_instruction_r.A_precision <= to_unsigned(0,fpu_next_instruction_r.A_precision'length); -- ???
                        end if;
                    elsif(fpu_set_P=register2_fpu_set_P_B) then
                        if(fpu_set_M=register2_fpu_set_M_VALUE) then
                            fpu_next_instruction_r.B_addr <= (others=>'0');
                            if(fpu_set_W=register2_fpu_set_W_FP16 ) then
                                fpu_next_instruction_r.B <= bus_writedata_in(fp16_t'length-1 downto 0) & "0000000000000000";
                            elsif(fpu_set_W=register2_fpu_set_W_ZFP16 ) then
                                fpu_next_instruction_r.B <= bus_writedata_in(fp12_t'length-1 downto 0) & "0000000000000000";
                            else
                                fpu_next_instruction_r.B <= bus_writedata_in;
                            end if;
                            fpu_next_instruction_r.B_by_value <= '1';
                        else
                            fpu_next_instruction_r.B_addr <= page_vm;
                            fpu_next_instruction_r.B <= (others=>'0');
                            fpu_next_instruction_r.B_by_value <= '0';
                        end if;
                        if(fpu_set_W=register2_fpu_set_W_INT8) then
                            fpu_next_instruction_r.B_precision <= to_unsigned(1,fpu_next_instruction_r.B_precision'length);                    
                        elsif(fpu_set_W=register2_fpu_set_W_FP16 or fpu_set_W=register2_fpu_set_W_ZFP16) then
                            fpu_next_instruction_r.B_precision <= to_unsigned(2,fpu_next_instruction_r.B_precision'length);
                        else
                            fpu_next_instruction_r.B_precision <= to_unsigned(4,fpu_next_instruction_r.B_precision'length);
                        end if;
                        fpu_next_instruction_r.B_enable <= '1';
                    elsif(fpu_set_P=register2_fpu_set_P_C) then
                        if(fpu_set_M=register2_fpu_set_M_VALUE) then
                            fpu_next_instruction_r.C_addr <= (others=>'0');
                            if(fpu_set_W=register2_fpu_set_W_FP16) then
                                fpu_next_instruction_r.C <= bus_writedata_in(fp16_t'length-1 downto 0) & "0000000000000000";
                            elsif(fpu_set_W=register2_fpu_set_W_ZFP16) then
                                fpu_next_instruction_r.C <= bus_writedata_in(fp12_t'length-1 downto 0) & "0000000000000000";
                            else
                                fpu_next_instruction_r.C <= bus_writedata_in(fp32_t'length-1 downto 0);
                            end if;
                            fpu_next_instruction_r.C_by_value <= '1';
                            fpu_next_instruction_r.C_pending <= '0';
                            fpu_next_instruction_r.C_enable <= '1';
                        else
                            fpu_next_instruction_r.C_addr <= page_vm;
                            fpu_next_instruction_r.C <= (others=>'0');
                            fpu_next_instruction_r.C_by_value <= '0';
                            fpu_next_instruction_r.C_pending <= '0';
                            fpu_next_instruction_r.C_enable <= '1';
                        end if;
                        if(fpu_set_W=register2_fpu_set_W_FP16 or fpu_set_W=register2_fpu_set_W_ZFP16) then
                            fpu_next_instruction_r.C_double <= '0';
                        else
                            fpu_next_instruction_r.C_double <= '1';
                        end if;
                    elsif(fpu_set_P=register2_fpu_set_P_C2) then
                        if(fpu_set_M=register2_fpu_set_M_VALUE) then
                            fpu_next_instruction_r.C2_addr <= (others=>'0');
                            if(fpu_set_W=register2_fpu_set_W_FP16) then
                                fpu_next_instruction_r.C2 <= bus_writedata_in(fp16_t'length-1 downto 0) & "0000000000000000";
                            elsif(fpu_set_W=register2_fpu_set_W_ZFP16) then
                                fpu_next_instruction_r.C2 <= bus_writedata_in(fp12_t'length-1 downto 0) & "0000000000000000";
                            else
                                fpu_next_instruction_r.C2 <= bus_writedata_in(fp32_t'length-1 downto 0);
                            end if;
                            fpu_next_instruction_r.C2_by_value <= '1';
                            fpu_next_instruction_r.C2_pending <= '0';
                            fpu_next_instruction_r.C2_enable <= '1';
                        else
                            fpu_next_instruction_r.C2_addr <= page_vm;
                            fpu_next_instruction_r.C2 <= (others=>'0');
                            fpu_next_instruction_r.C2_by_value <= '0';
                            fpu_next_instruction_r.C2_pending <= '0';
                            fpu_next_instruction_r.C2_enable <= '1';
                        end if;
                        if(fpu_set_W=register2_fpu_set_W_FP16 or fpu_set_W=register2_fpu_set_W_ZFP16) then
                            fpu_next_instruction_r.C2_double <= '0';
                        else
                            fpu_next_instruction_r.C2_double <= '1';
                        end if;
                    elsif(fpu_set_P=register2_fpu_set_P_X) then
                        if(fpu_set_M = register2_fpu_set_M_VALUE) then
                            fpu_next_instruction_r.X_addr <= (others=>'0');
                            if(fpu_set_W=register2_fpu_set_W_FP16) then
                                fpu_next_instruction_r.X <= bus_writedata_in(fp16_t'length-1 downto 0) & "0000000000000000";
                            elsif(fpu_set_W=register2_fpu_set_W_ZFP16) then
                                fpu_next_instruction_r.X <= bus_writedata_in(fp12_t'length-1 downto 0) & "0000000000000000";
                            else
                                fpu_next_instruction_r.X <= bus_writedata_in(fp32_t'length-1 downto 0);
                            end if;
                            fpu_next_instruction_r.X_enable <= '1';
                            fpu_next_instruction_r.X_by_value <= '1';
                        else
                            fpu_next_instruction_r.X_addr <= page_vm;
                            fpu_next_instruction_r.X <= (others=>'0');
                            fpu_next_instruction_r.X_enable <= '1';
                            fpu_next_instruction_r.X_by_value <= '0';
                        end if;
                        if(fpu_set_W = register2_fpu_set_W_FP16 or fpu_set_W = register2_fpu_set_W_ZFP16) then
                            fpu_next_instruction_r.X_double <= '0';
                        else
                            fpu_next_instruction_r.X_double <= '1';
                        end if;
                        fpu_next_instruction_r.X_type <= fpu_set_W;
                    elsif(fpu_set_P=register2_fpu_set_P_Y) then
                        if(fpu_set_M = register2_fpu_set_M_VALUE) then
                            fpu_next_instruction_r.Y_addr <= (others=>'0');
                            if(fpu_set_W=register2_fpu_set_W_FP16) then
                                fpu_next_instruction_r.Y <= bus_writedata_in(fp16_t'length-1 downto 0) & "0000000000000000";
                            elsif(fpu_set_W=register2_fpu_set_W_ZFP16) then
                                fpu_next_instruction_r.Y <= bus_writedata_in(fp12_t'length-1 downto 0) & "0000000000000000";
                            else
                                fpu_next_instruction_r.Y <= bus_writedata_in(fp32_t'length-1 downto 0);
                            end if;
                            fpu_next_instruction_r.Y_enable <= '1';
                            fpu_next_instruction_r.Y_by_value <= '1';
                        else
                            fpu_next_instruction_r.Y_addr <= page_vm;
                            fpu_next_instruction_r.Y <= (others=>'0');
                            fpu_next_instruction_r.Y_enable <= '1';
                            fpu_next_instruction_r.Y_by_value <= '0';
                        end if;
                        fpu_next_instruction_r.Y_type <= fpu_set_W;
                        if(fpu_set_W = register2_fpu_set_W_FP16 or fpu_set_W = register2_fpu_set_W_ZFP16) then
                            fpu_next_instruction_r.Y_double <= '0';
                        else
                            fpu_next_instruction_r.Y_double <= '1';
                        end if;
                    elsif(fpu_set_P=register2_fpu_set_P_CNT) then
                        fpu_next_instruction_r.CNT <= unsigned(bus_writedata_in(sram_depth_c-1 downto 0));
                    end if;
                end if;
            end if;

            if(cmd_fifo_rd/="00") then
                running_r <= '1';
                busy_v := '1';
                step_r <= (others=>'0');
                fpu_instruction_r <= fpu_instruction;
            end if;

            if(exe='1' and eof='1' and (fpu_instruction_r.LAST='1' or fpu_instruction_r.FAST='0')) then
                write_flush_r <= '1';
            elsif(fpu_eof='1' and fpu_write='1' and (fpu_last='1' or fpu_fast='0')) then
                write_flush_r <= '0';
            end if;

            if(cmd_fifo_we/="00") then
                -- Clear for next FPU instruction
                fpu_next_instruction_r.B_enable <= '0';
                fpu_next_instruction_r.C_enable <= '0';
                fpu_next_instruction_r.C2_enable <= '0';
                fpu_next_instruction_r.X_enable <= '0';
                fpu_next_instruction_r.Y_enable <= '0';
            end if;

            if(fpu_write='1' and fpu_eof='1' and fpu_last='1') then
                busy_v := '0';
            end if;

            ------
            -- Process read commands
            ------

            if(bus_read_in='1') then
                -- Process read commands
                if(rregno = to_unsigned(register_fpu_get_status_c,register_t'length)) then
                    rden_r <= '1';
                    rresp_r(rresp_r'length-1 downto 1) <= (others=>'0');
                    rresp_r(0) <= busy_r;
                else
                    rden_r <= '0';
                end if;
            else
                rden_r <= '0';
            end if;
            busy_rr <= busy_r;
            busy_r <= busy_v;
            fpu_exe_rr <= fpu_exe_r;
            fpu_exe_r <= fpu_exe_v;
            fpu_vm_r <= fpu_vm_v;
            fpu_exe_pending_r <= fpu_exe_pending_v;
            fpu_busy_vm_r(0) <= ((busy_v or busy_r or fpu_exe_v or fpu_exe_r) and (not fpu_vm_v)) or fpu_exe_pending_v(0);
            fpu_busy_vm_r(1) <= ((busy_v or busy_r or fpu_exe_v or fpu_exe_r) and (fpu_vm_v)) or fpu_exe_pending_v(1);

        end if;
    end if;
end process;
END behavior;
