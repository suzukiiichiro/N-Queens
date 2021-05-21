-- EMACS settings: -*-  tab-width: 2; indent-tabs-mode: t -*-
-- vim: tabstop=2:shiftwidth=2:noexpandtab
-- kate: tab-width 2; replace-tabs off; indent-width 2;
-------------------------------------------------------------------------------
-- This file is part of the Queens@TUD solver suite
-- for enumerating and counting the solutions of an N-Queens Puzzle.
--
-- Copyright (C) 2008-2016
--      Thomas B. Preusser <thomas.preusser@utexas.edu>
-------------------------------------------------------------------------------
-- This testbench is free software: you can redistribute it and/or modify
-- it under the terms of the GNU Affero General Public License as published
-- by the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU Affero General Public License for more details.
--
-- You should have received a copy of the GNU Affero General Public License
-- along with this design.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------------------

entity queens_slice_tb is
end queens_slice_tb;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

architecture tb of queens_slice_tb is

  constant L : natural := 2;  -- pre-placed rings:
                              --   testbench only supports 2 currently

  type tTest is record
                  cnt : positive;
                  bh  : positive;
                  bv  : positive;
                  bu  : positive;
                  bd  : positive;
                end record tTest;
  type tTests is array (natural range<>) of tTest;

  function selectTests(s : positive) return tTests is
    constant TESTS_8 : tTests := (
      (      1 , 10,12,27,41),
      (      1 , 10,12,82,25),
      (      1 , 10,3,25,82),
      (      1 , 10,3,41,27),
      (      1 , 10,6,67,78),
      (      1 , 10,6,78,67),
      (      1 , 11,11,114,106),
      (      1 , 11,11,116,102),
      (      1 , 11,11,22,103),
      (      1 , 11,11,23,102),
      (      1 , 11,11,39,106),
      (      1 , 11,11,49,108),
      (      1 , 11,11,52,103),
      (      1 , 11,11,70,108),
      (      1 , 11,13,102,116),
      (      1 , 11,13,102,23),
      (      1 , 11,13,103,22),
      (      1 , 11,13,103,52),
      (      1 , 11,13,106,114),
      (      1 , 11,13,106,39),
      (      1 , 11,13,108,49),
      (      1 , 11,13,108,70),
      (      1 , 11,14,87,82),
      (      1 , 11,7,82,87),
      (      1 , 12,10,108,41),
      (      1 , 12,10,37,25),
      (      1 , 12,5,25,37),
      (      1 , 12,5,41,108),
      (      1 , 13,11,115,22),
      (      1 , 13,11,115,52),
      (      1 , 13,11,27,49),
      (      1 , 13,11,27,70),
      (      1 , 13,11,43,114),
      (      1 , 13,11,43,39),
      (      1 , 13,11,51,116),
      (      1 , 13,11,51,23),
      (      1 , 13,13,114,43),
      (      1 , 13,13,116,51),
      (      1 , 13,13,22,115),
      (      1 , 13,13,23,51),
      (      1 , 13,13,39,43),
      (      1 , 13,13,49,27),
      (      1 , 13,13,52,115),
      (      1 , 13,13,70,27),
      (      1 , 13,14,37,117),
      (      1 , 13,7,117,37),
      (      1 , 14,11,117,82),
      (      1 , 14,13,82,117),
      (      1 , 15,15,119,99),
      (      1 , 15,15,99,119),
      (      1 , 2,2,25,80),
      (      1 , 2,2,76,80),
      (      1 , 2,4,80,25),
      (      1 , 2,4,80,76),
      (      1 , 3,10,74,27),
      (      1 , 3,10,76,82),
      (      1 , 3,5,27,74),
      (      1 , 3,5,82,76),
      (      1 , 4,2,5,25),
      (      1 , 4,2,5,76),
      (      1 , 4,4,25,5),
      (      1 , 4,4,76,5),
      (      1 , 5,12,74,108),
      (      1 , 5,12,76,37),
      (      1 , 5,3,108,74),
      (      1 , 5,3,37,76),
      (      1 , 5,6,57,97),
      (      1 , 5,6,97,57),
      (      1 , 6,10,57,67),
      (      1 , 6,10,97,78),
      (      1 , 6,5,67,57),
      (      1 , 6,5,78,97),
      (      1 , 6,9,105,20),
      (      1 , 6,9,106,88),
      (      1 , 6,9,13,43),
      (      1 , 6,9,20,105),
      (      1 , 6,9,20,75),
      (      1 , 6,9,43,13),
      (      1 , 6,9,75,20),
      (      1 , 6,9,88,106),
      (      1 , 7,11,37,87),
      (      1 , 7,13,87,37),
      (      1 , 9,6,105,20),
      (      1 , 9,6,106,13),
      (      1 , 9,6,13,106),
      (      1 , 9,6,20,105),
      (      1 , 9,6,20,75),
      (      1 , 9,6,43,88),
      (      1 , 9,6,75,20),
      (      1 , 9,6,88,43)
    );
    constant TESTS_9 : tTests := (
      (      1 , 10,12,243,133),
      (      1 , 10,12,243,257),
      (      1 , 10,12,257,414),
      (      1 , 10,12,322,414),
      (      1 , 10,17,108,325),
      (      1 , 10,17,325,108),
      (      1 , 10,17,365,68),
      (      1 , 10,17,68,365),
      (      1 , 10,20,147,41),
      (      1 , 10,20,149,281),
      (      1 , 10,20,275,344),
      (      1 , 10,20,296,402),
      (      1 , 10,20,305,338),
      (      1 , 10,20,53,401),
      (      1 , 10,5,281,149),
      (      1 , 10,5,338,305),
      (      1 , 10,5,344,275),
      (      1 , 10,5,401,53),
      (      1 , 10,5,402,296),
      (      1 , 10,5,41,147),
      (      1 , 10,6,133,243),
      (      1 , 10,6,257,243),
      (      1 , 10,6,414,257),
      (      1 , 10,6,414,322),
      (      1 , 11,13,193,247),
      (      1 , 11,13,199,181),
      (      1 , 11,13,203,226),
      (      1 , 11,13,449,181),
      (      1 , 11,13,451,118),
      (      1 , 11,14,167,211),
      (      1 , 11,14,211,167),
      (      1 , 11,14,211,417),
      (      1 , 11,14,417,211),
      (      1 , 11,19,452,109),
      (      1 , 11,22,118,451),
      (      1 , 11,22,181,199),
      (      1 , 11,22,181,449),
      (      1 , 11,22,226,203),
      (      1 , 11,22,247,193),
      (      1 , 11,25,109,452),
      (      1 , 12,10,133,414),
      (      1 , 12,10,257,414),
      (      1 , 12,10,414,133),
      (      1 , 12,10,414,257),
      (      1 , 12,17,300,305),
      (      1 , 12,17,305,300),
      (      1 , 12,17,306,37),
      (      1 , 12,17,37,306),
      (      1 , 12,18,260,437),
      (      1 , 12,18,284,389),
      (      1 , 12,18,388,374),
      (      1 , 12,20,237,259),
      (      1 , 12,5,259,237),
      (      1 , 12,9,374,388),
      (      1 , 12,9,389,284),
      (      1 , 12,9,437,260),
      (      1 , 13,11,262,247),
      (      1 , 13,11,263,181),
      (      1 , 13,11,391,118),
      (      1 , 13,11,422,226),
      (      1 , 13,11,454,181),
      (      1 , 13,19,138,174),
      (      1 , 13,19,214,71),
      (      1 , 13,19,313,420),
      (      1 , 13,19,342,353),
      (      1 , 13,21,133,379),
      (      1 , 13,21,203,43),
      (      1 , 13,21,205,331),
      (      1 , 13,21,331,205),
      (      1 , 13,21,334,355),
      (      1 , 13,21,355,334),
      (      1 , 13,21,379,133),
      (      1 , 13,21,43,203),
      (      1 , 13,25,174,138),
      (      1 , 13,25,353,342),
      (      1 , 13,25,420,313),
      (      1 , 13,25,71,214),
      (      1 , 13,26,118,391),
      (      1 , 13,26,181,263),
      (      1 , 13,26,181,454),
      (      1 , 13,26,226,422),
      (      1 , 13,26,247,262),
      (      1 , 13,28,171,139),
      (      1 , 13,28,323,477),
      (      1 , 13,28,67,491),
      (      1 , 13,7,139,171),
      (      1 , 13,7,477,323),
      (      1 , 13,7,491,67),
      (      1 , 14,11,267,211),
      (      1 , 14,11,406,167),
      (      1 , 14,11,406,417),
      (      1 , 14,11,458,211),
      (      1 , 14,19,421,404),
      (      1 , 14,19,83,331),
      (      1 , 14,25,331,83),
      (      1 , 14,25,404,421),
      (      1 , 14,26,167,406),
      (      1 , 14,26,211,267),
      (      1 , 14,26,211,458),
      (      1 , 14,26,417,406),
      (      1 , 1,4,49,73),
      (      1 , 1,4,73,49),
      (      1 , 15,23,455,359),
      (      1 , 15,29,359,455),
      (      1 , 16,4,280,292),
      (      1 , 16,4,292,280),
      (      1 , 17,10,108,325),
      (      1 , 17,10,325,108),
      (      1 , 17,10,365,68),
      (      1 , 17,10,68,365),
      (      1 , 17,12,105,305),
      (      1 , 17,12,153,37),
      (      1 , 17,12,281,300),
      (      1 , 17,12,328,306),
      (      1 , 17,6,300,281),
      (      1 , 17,6,305,105),
      (      1 , 17,6,306,328),
      (      1 , 17,6,37,153),
      (      1 , 18,12,113,389),
      (      1 , 18,12,65,437),
      (      1 , 18,12,67,374),
      (      1 , 18,20,165,25),
      (      1 , 18,20,165,49),
      (      1 , 18,24,101,356),
      (      1 , 18,24,149,88),
      (      1 , 18,3,356,101),
      (      1 , 18,3,88,149),
      (      1 , 18,5,25,165),
      (      1 , 18,5,49,165),
      (      1 , 18,6,374,67),
      (      1 , 18,6,389,113),
      (      1 , 18,6,437,65),
      (      1 , 19,11,71,109),
      (      1 , 19,13,162,174),
      (      1 , 19,13,213,353),
      (      1 , 19,13,214,71),
      (      1 , 19,13,313,420),
      (      1 , 19,14,331,404),
      (      1 , 19,14,404,331),
      (      1 , 19,21,181,204),
      (      1 , 19,21,204,181),
      (      1 , 19,22,174,162),
      (      1 , 19,22,353,213),
      (      1 , 19,22,420,313),
      (      1 , 19,22,71,214),
      (      1 , 19,26,109,71),
      (      1 , 20,10,281,338),
      (      1 , 20,10,338,281),
      (      1 , 20,10,344,401),
      (      1 , 20,10,401,344),
      (      1 , 20,10,402,41),
      (      1 , 20,10,41,402),
      (      1 , 20,12,366,259),
      (      1 , 20,18,330,25),
      (      1 , 20,18,330,49),
      (      1 , 20,6,259,366),
      (      1 , 20,9,25,330),
      (      1 , 20,9,49,330),
      (      1 , 21,13,229,355),
      (      1 , 21,13,322,379),
      (      1 , 21,13,358,331),
      (      1 , 21,13,397,334),
      (      1 , 21,13,421,205),
      (      1 , 21,13,422,43),
      (      1 , 21,13,424,203),
      (      1 , 21,13,445,133),
      (      1 , 21,19,102,181),
      (      1 , 21,19,346,204),
      (      1 , 21,22,133,445),
      (      1 , 21,22,203,424),
      (      1 , 21,22,205,421),
      (      1 , 21,22,331,358),
      (      1 , 21,22,334,397),
      (      1 , 21,22,355,229),
      (      1 , 21,22,379,322),
      (      1 , 21,22,43,422),
      (      1 , 21,25,181,102),
      (      1 , 21,25,204,346),
      (      1 , 22,11,142,203),
      (      1 , 22,11,220,451),
      (      1 , 22,11,346,199),
      (      1 , 22,11,346,449),
      (      1 , 22,11,478,193),
      (      1 , 22,19,234,162),
      (      1 , 22,19,269,213),
      (      1 , 22,19,452,214),
      (      1 , 22,19,75,313),
      (      1 , 22,21,229,397),
      (      1 , 22,21,322,445),
      (      1 , 22,21,358,421),
      (      1 , 22,21,397,229),
      (      1 , 22,21,421,358),
      (      1 , 22,21,422,424),
      (      1 , 22,21,424,422),
      (      1 , 22,21,445,322),
      (      1 , 22,25,162,234),
      (      1 , 22,25,213,269),
      (      1 , 22,25,214,452),
      (      1 , 22,25,313,75),
      (      1 , 22,26,193,478),
      (      1 , 22,26,199,346),
      (      1 , 22,26,203,142),
      (      1 , 22,26,449,346),
      (      1 , 22,26,451,220),
      (      1 , 22,28,375,389),
      (      1 , 22,28,418,426),
      (      1 , 22,28,431,388),
      (      1 , 22,7,388,431),
      (      1 , 22,7,389,375),
      (      1 , 22,7,426,418),
      (      1 , 23,15,455,359),
      (      1 , 23,23,103,423),
      (      1 , 23,23,460,423),
      (      1 , 23,29,423,103),
      (      1 , 23,29,423,460),
      (      1 , 23,30,359,455),
      (      1 , 2,4,138,42),
      (      1 , 24,18,332,356),
      (      1 , 24,18,338,88),
      (      1 , 2,4,42,138),
      (      1 , 24,9,356,332),
      (      1 , 24,9,88,338),
      (      1 , 25,11,364,452),
      (      1 , 25,13,234,138),
      (      1 , 25,13,269,342),
      (      1 , 25,13,452,214),
      (      1 , 25,13,75,313),
      (      1 , 25,14,421,83),
      (      1 , 25,14,83,421),
      (      1 , 25,21,102,346),
      (      1 , 25,21,346,102),
      (      1 , 25,22,138,234),
      (      1 , 25,22,214,452),
      (      1 , 25,22,313,75),
      (      1 , 25,22,342,269),
      (      1 , 25,26,452,364),
      (      1 , 26,13,142,422),
      (      1 , 26,13,220,391),
      (      1 , 26,13,346,263),
      (      1 , 26,13,346,454),
      (      1 , 26,13,478,262),
      (      1 , 26,14,267,406),
      (      1 , 26,14,406,267),
      (      1 , 26,14,406,458),
      (      1 , 26,14,458,406),
      (      1 , 26,19,364,71),
      (      1 , 26,22,262,478),
      (      1 , 26,22,263,346),
      (      1 , 26,22,391,220),
      (      1 , 26,22,422,142),
      (      1 , 26,22,454,346),
      (      1 , 26,25,71,364),
      (      1 , 27,27,103,103),
      (      1 , 27,27,103,460),
      (      1 , 27,27,108,455),
      (      1 , 27,27,198,365),
      (      1 , 27,27,365,198),
      (      1 , 27,27,455,108),
      (      1 , 27,27,460,103),
      (      1 , 27,27,460,460),
      (      1 , 28,13,388,491),
      (      1 , 28,13,389,477),
      (      1 , 28,13,426,139),
      (      1 , 28,22,139,426),
      (      1 , 28,22,477,389),
      (      1 , 28,22,491,388),
      (      1 , 29,15,461,455),
      (      1 , 29,23,459,103),
      (      1 , 29,23,459,460),
      (      1 , 29,29,103,459),
      (      1 , 29,29,460,459),
      (      1 , 29,30,455,461),
      (      1 , 30,23,461,455),
      (      1 , 30,29,455,461),
      (      1 , 3,18,52,149),
      (      1 , 3,18,77,101),
      (      1 , 3,9,101,77),
      (      1 , 3,9,149,52),
      (      1 , 4,1,280,73),
      (      1 , 4,1,292,49),
      (      1 , 4,16,49,292),
      (      1 , 4,16,73,280),
      (      1 , 4,2,162,42),
      (      1 , 4,2,168,138),
      (      1 , 4,8,138,168),
      (      1 , 4,8,42,162),
      (      1 , 5,10,147,296),
      (      1 , 5,10,149,305),
      (      1 , 5,10,275,53),
      (      1 , 5,10,296,147),
      (      1 , 5,10,305,149),
      (      1 , 5,10,53,275),
      (      1 , 5,12,385,237),
      (      1 , 5,18,280,165),
      (      1 , 5,18,304,165),
      (      1 , 5,6,237,385),
      (      1 , 5,9,165,280),
      (      1 , 5,9,165,304),
      (      1 , 6,10,243,257),
      (      1 , 6,10,243,322),
      (      1 , 6,10,257,243),
      (      1 , 6,10,322,243),
      (      1 , 6,17,105,281),
      (      1 , 6,17,153,328),
      (      1 , 6,17,281,105),
      (      1 , 6,17,328,153),
      (      1 , 6,18,221,67),
      (      1 , 6,18,323,113),
      (      1 , 6,18,347,65),
      (      1 , 6,20,385,366),
      (      1 , 6,5,366,385),
      (      1 , 6,9,113,323),
      (      1 , 6,9,65,347),
      (      1 , 6,9,67,221),
      (      1 , 7,13,375,323),
      (      1 , 7,13,418,171),
      (      1 , 7,13,431,67),
      (      1 , 7,22,171,418),
      (      1 , 7,22,323,375),
      (      1 , 7,22,67,431),
      (      1 , 8,4,162,168),
      (      1 , 8,4,168,162),
      (      1 , 9,12,221,388),
      (      1 , 9,12,323,284),
      (      1 , 9,12,347,260),
      (      1 , 9,20,280,330),
      (      1 , 9,20,304,330),
      (      1 , 9,24,52,338),
      (      1 , 9,24,77,332),
      (      1 , 9,3,332,77),
      (      1 , 9,3,338,52),
      (      1 , 9,5,330,280),
      (      1 , 9,5,330,304),
      (      1 , 9,6,260,347),
      (      1 , 9,6,284,323),
      (      1 , 9,6,388,221)
    );
    constant TESTS_11 : tTests := (
      (      1 , 18,5,4424,167 ),
      (      1 , 18,5,4676,5280 ),
      (      1 , 18,5,6736,1160 ),
      (      2 , 18,65,4256,676 ),
      (      2 , 18,65,676,4256 ),
      (      1 , 18,66,4232,708 ),
      (      1 , 18,66,568,2193 ),
      (      1 , 18,66,660,4177 ),
      (      1 , 18,66,676,4672 ),
      (      2 , 20,34,2570,680 ),
      (      1 , 20,34,4369,5285 ),
      (      1 , 20,34,4529,5125 ),
      (      1 , 20,34,4618,708 ),
      (      1 , 20,34,5125,4529 ),
      (      1 , 20,34,5285,4369 ),
      (      2 , 20,34,680,2570 ),
      (      1 , 20,3,4688,167 ),
      (      1 , 20,3,4688,646 ),
      (      1 , 20,34,708,4618 ),
      (      1 , 97,26,6409,2404 ),
      (      1 , 97,28,4882,7330 ),
      (      2 , 26,81,2374,4379 ),
      (      1 , 26,81,2438,4395 ),
      (      1 , 26,81,2441,2439 ),
      (      1 , 26,81,3336,6371 ),
      (      1 , 41,84,2318,203 ),
      (      2 , 41,84,2326,4427 ),
      (      1 , 44,81,6371,534 ),
      (      1 , 44,81,6801,3122 ),
      (      2 , 44,81,6929,3154 ),
      (      1 , 44,81,7218,4658 ),
      (      1 , 44,81,741,1038 ),
      (      1 , 65,24,2145,2340 ),
      (      2 , 65,36,4256,1192 ),
      (      2 , 65,36,676,161 ),
      (      1 , 65,40,2097,2224 ),
      (      1 , 65,40,418,4482 ),
      (      1 , 66,10,1232,1569 )
    );
    constant TESTS_12 : tTests := (
      (      2 , 75,165,787,4891 ),
      (      2 , 76,196,10914,21521 ),
      (      2 , 76,35,21521,10914 ),
      (      2 , 77,170,10531,5661 ),
      (      2 , 77,85,5661,10531 ),
      (      2 , 80,12,12648,20489 ),
      (      3 , 80,12,24804,18438 ),
      (      2 , 80,20,26784,22660 ),
      (      2 , 80,40,22660,26784 ),
      (      3 , 80,48,18438,24804 ),
      (      2 , 80,48,20489,12648 ),
      (      2 , 82,104,13857,18701 ),
      (      2 , 82,168,7316,16458 ),
      (      2 , 82,21,16458,7316 ),
      (      2 , 82,22,18701,13857 ),
      (      2 , 82,42,22532,18074 ),
      (      2 , 82,74,13872,18693 ),
      (      2 , 82,74,13872,20553 ),
      (      2 , 82,82,18693,13872 ),
      (      2 , 82,82,20553,13872 ),
      (      2 , 82,84,18074,22532 ),
      (      2 , 84,11,20805,2586 ),
      (      2 , 84,137,22820,10337 ),
      (      2 , 84,145,10337,22820 ),
      (      2 , 84,162,9377,27684 ),
      (      2 , 84,208,2586,20805 ),
      (      2 , 84,69,27684,9377 ),
      (      2 , 84,74,22532,11441 ),
      (      2 , 84,82,11441,22532 ),
      (      2 , 85,108,22042,11292 ),
      (      2 , 85,178,10531,23604 ),
      (      2 , 85,54,11292,22042 ),
      (      2 , 85,77,23604,10531 ),
      (      2 , 88,100,5149,10409 ),
      (      2 , 88,38,10409,5149 ),
      (      2 , 90,195,20805,9778 ),
      (      2 , 90,195,9778,20805 ),
      (      2 , 98,104,21777,11304 ),
      (      2 , 98,148,4441,25106 ),
      (      2 , 98,22,11304,21777 ),
      (      2 , 98,41,25106,4441 ),
      (      2 , 98,44,21785,18474 ),
      (      2 , 98,52,18474,21785 ),
      (      1 , 99,116,19754,21529 ),
      (      1 , 99,120,15379,18740 ),
      (      1 , 99,120,6419,27700 ),
      (      1 , 99,150,26154,17701 ),
      (      1 , 99,150,27696,6449 ),
      (      1 , 99,150,28977,9769 ),
      (      1 , 99,156,18701,23122 ),
      (      1 , 99,165,17225,18713 ),
      (      1 , 99,165,18713,17225 ),
      (      1 , 99,170,25393,13353 ),
      (      1 , 99,172,17677,27218 ),
      (      1 , 99,172,22809,19017 ),
      (      1 , 99,172,23576,27233 ),
      (      1 , 99,180,13105,25626 ),
      (      1 , 99,180,17677,27192 ),
      (      1 , 99,180,19740,25129 ),
      (      1 , 99,198,17177,17993 ),
      (      1 , 99,198,17177,18737 ),
      (      1 , 99,204,17176,27700 ),
      (      1 , 99,204,4889,11313 ),
      (      1 , 99,212,11801,4401 ),
      (      1 , 99,212,11824,21554 ),
      (      1 , 99,212,19760,21546 ),
      (      1 , 99,212,3608,20787 ),
      (      1 , 99,216,22050,19021 ),
      (      1 , 99,27,19021,22050 ),
      (      1 , 99,30,18740,15379 ),
      (      1 , 99,30,27700,6419 ),
      (      1 , 99,43,20787,3608 ),
      (      1 , 99,43,21546,19760 ),
      (      1 , 99,43,21554,11824 ),
      (      1 , 99,43,4401,11801 ),
      (      1 , 99,45,25129,19740 ),
      (      1 , 99,45,25626,13105 )
    );
  begin
    case s is
      when  8 => return  TESTS_8;
      when  9 => return  TESTS_9;
      when 11 => return  TESTS_11;
      when 12 => return  TESTS_12;
      when others => null;
    end case;
    report "Unsupported problem size "&integer'image(s)&'.'
      severity failure;
  end;

  component queens_slice
    generic (
      N : positive;                     -- size of field
      L : positive                      -- number of preplaced columns
    );
    port (
      clk   : IN  std_logic;
      rst   : IN  std_logic;
      start : IN  std_logic;
      BH_l  : IN  std_logic_vector(0 to N-2*L-1);
      BU_l  : IN  std_logic_vector(0 to 2*N-4*L-2);
      BD_l  : IN  std_logic_vector(0 to 2*N-4*L-2);
      BV_l  : IN  std_logic_vector(0 to N-2*L-1);
      sol   : OUT std_logic;
      done  : OUT std_logic
    );
  end component;

   -- Clock period definitions
  constant clk_period : time := 10 ns;

begin

	genSizes: for s in 8 to 12 generate
		genFilter: if s /= 10 generate
			constant TESTS : tTests := selectTests(s);

			--Inputs
			signal clk   : std_logic;
			signal rst   : std_logic;
			signal start : std_logic;
			signal bh    : std_logic_vector(0 to s-2*L-1);
			signal bv    : std_logic_vector(0 to s-2*L-1);
			signal bu    : std_logic_vector(0 to 2*s-4*L-2);
			signal bd    : std_logic_vector(0 to 2*s-4*L-2);

			--Outputs
			signal sol  : std_logic;
			signal done : std_logic;

			-- Test Control
			signal nxt : boolean;
		begin

			dut: queens_slice
				generic map (
					N => s,
					L => L
				)
				port map (
					clk   => clk,
					rst   => rst,
					start => start,
					BH_l  => bh,
					BV_l  => bv,
					BU_l  => bu,
					BD_l  => bd,
					sol   => sol,
					done  => done
				);

			-- Stimuli
			process
				procedure cycle is
				begin
					clk <= '0';
					wait for clk_period/2;
					clk <= '1';
					wait for clk_period/2;
				end;
			begin
				rst   <= '1';
				cycle;
				rst   <= '0';
				start <= '0';
				cycle;

				for i in TESTS'range loop

					bh    <= std_logic_vector(to_unsigned(TESTS(i).bh, bh'length));
					bv    <= std_logic_vector(to_unsigned(TESTS(i).bv, bv'length));
					bu    <= std_logic_vector(to_unsigned(TESTS(i).bu, bu'length));
					bd    <= std_logic_vector(to_unsigned(TESTS(i).bd, bd'length));
					start <= '1';
					cycle;

					bh    <= (others => '-');
					bv    <= (others => '-');
					bu    <= (others => '-');
					bd    <= (others => '-');
					start <= '0';

					loop
						cycle;
						exit when nxt;
					end loop;
				end loop;

				wait;  -- forever
			end process;

			-- Checker
			process
				variable err : natural;
				variable cnt : natural;
			begin
				err := 0;
				for i in TESTS'range loop
					nxt <= true;
					wait until rising_edge(clk) and start = '1';
					nxt <= false;

					cnt := 0;
					loop
						wait until rising_edge(clk);
						if sol = '1' then
							cnt := cnt + 1;
						end if;
						exit when done = '1';
					end loop;
					if cnt /= TESTS(i).cnt then
						report "Result mismatch in test case #"&integer'image(i)&": "&
							integer'image(TESTS(i).cnt)&" -> "&integer'image(cnt)
							severity error;
						err := err + 1;
					end if;
				end loop;

				if err = 0 then
					report "Test [N="&integer'image(s)&", L="&integer'image(L)&"] completed successfully." severity note;
				else
					report "Test [N="&integer'image(s)&", L="&integer'image(L)&"] completed with "&integer'image(err)&" ERRORS." severity note;
				end if;

			end process;

		end generate;
	end generate;

end tb;
