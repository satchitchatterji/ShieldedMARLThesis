% Waterworld Shield
% actions: up, down, left, right, none
% sensors: 30 obstacles, 30 barriers, 30 food, 30 poison, 30 pursuer, 2 collisions

% actions
action(0)::action(up);
action(1)::action(down);
action(2)::action(left);
action(3)::action(right);
action(4)::action(upright);
action(5)::action(downright);
action(6)::action(upleft);
action(7)::action(downleft);
action(8)::action(none).

% 32 sensors for obstacles
sensor_value(0)::obstacle(0).
sensor_value(1)::obstacle(1).
sensor_value(2)::obstacle(2).
sensor_value(3)::obstacle(3).
sensor_value(4)::obstacle(4).
sensor_value(5)::obstacle(5).
sensor_value(6)::obstacle(6).
sensor_value(7)::obstacle(7).
sensor_value(8)::obstacle(8).
sensor_value(9)::obstacle(9).
sensor_value(10)::obstacle(10).
sensor_value(11)::obstacle(11).
sensor_value(12)::obstacle(12).
sensor_value(13)::obstacle(13).
sensor_value(14)::obstacle(14).
sensor_value(15)::obstacle(15).
sensor_value(16)::obstacle(16).
sensor_value(17)::obstacle(17).
sensor_value(18)::obstacle(18).
sensor_value(19)::obstacle(19).
sensor_value(20)::obstacle(20).
sensor_value(21)::obstacle(21).
sensor_value(22)::obstacle(22).
sensor_value(23)::obstacle(23).
sensor_value(24)::obstacle(24).
sensor_value(25)::obstacle(25).
sensor_value(26)::obstacle(26).
sensor_value(27)::obstacle(27).
sensor_value(28)::obstacle(28).
sensor_value(29)::obstacle(29).
sensor_value(30)::obstacle(30).
sensor_value(31)::obstacle(31).

% 32 sensors for barriers
sensor_value(32)::barrier(0).
sensor_value(33)::barrier(1).
sensor_value(34)::barrier(2).
sensor_value(35)::barrier(3).
sensor_value(36)::barrier(4).
sensor_value(37)::barrier(5).
sensor_value(38)::barrier(6).
sensor_value(39)::barrier(7).
sensor_value(40)::barrier(8).
sensor_value(41)::barrier(9).
sensor_value(42)::barrier(10).
sensor_value(43)::barrier(11).
sensor_value(44)::barrier(12).
sensor_value(45)::barrier(13).
sensor_value(46)::barrier(14).
sensor_value(47)::barrier(15).
sensor_value(48)::barrier(16).
sensor_value(49)::barrier(17).
sensor_value(50)::barrier(18).
sensor_value(51)::barrier(19).
sensor_value(52)::barrier(20).
sensor_value(53)::barrier(21).
sensor_value(54)::barrier(22).
sensor_value(55)::barrier(23).
sensor_value(56)::barrier(24).
sensor_value(57)::barrier(25).
sensor_value(58)::barrier(26).
sensor_value(59)::barrier(27).
sensor_value(60)::barrier(28).
sensor_value(61)::barrier(29).
sensor_value(62)::barrier(30).
sensor_value(63)::barrier(31).

% 32 sensors for food
sensor_value(64)::food(0).
sensor_value(65)::food(1).
sensor_value(66)::food(2).
sensor_value(67)::food(3).
sensor_value(68)::food(4).
sensor_value(69)::food(5).
sensor_value(70)::food(6).
sensor_value(71)::food(7).
sensor_value(72)::food(8).
sensor_value(73)::food(9).
sensor_value(74)::food(10).
sensor_value(75)::food(11).
sensor_value(76)::food(12).
sensor_value(77)::food(13).
sensor_value(78)::food(14).
sensor_value(79)::food(15).
sensor_value(80)::food(16).
sensor_value(81)::food(17).
sensor_value(82)::food(18).
sensor_value(83)::food(19).
sensor_value(84)::food(20).
sensor_value(85)::food(21).
sensor_value(86)::food(22).
sensor_value(87)::food(23).
sensor_value(88)::food(24).
sensor_value(89)::food(25).
sensor_value(90)::food(26).
sensor_value(91)::food(27).
sensor_value(92)::food(28).
sensor_value(93)::food(29).
sensor_value(94)::food(30).
sensor_value(95)::food(31).

% 32 sensors for poison
sensor_value(96)::poison(0).
sensor_value(97)::poison(1).
sensor_value(98)::poison(2).
sensor_value(99)::poison(3).
sensor_value(100)::poison(4).
sensor_value(101)::poison(5).
sensor_value(102)::poison(6).
sensor_value(103)::poison(7).
sensor_value(104)::poison(8).
sensor_value(105)::poison(9).
sensor_value(106)::poison(10).
sensor_value(107)::poison(11).
sensor_value(108)::poison(12).
sensor_value(109)::poison(13).
sensor_value(110)::poison(14).
sensor_value(111)::poison(15).
sensor_value(112)::poison(16).
sensor_value(113)::poison(17).
sensor_value(114)::poison(18).
sensor_value(115)::poison(19).
sensor_value(116)::poison(20).
sensor_value(117)::poison(21).
sensor_value(118)::poison(22).
sensor_value(119)::poison(23).
sensor_value(120)::poison(24).
sensor_value(121)::poison(25).
sensor_value(122)::poison(26).
sensor_value(123)::poison(27).
sensor_value(124)::poison(28).
sensor_value(125)::poison(29).
sensor_value(126)::poison(30).
sensor_value(127)::poison(31).

% 32 sensors for pursuer
sensor_value(128)::pursuer(0).
sensor_value(129)::pursuer(1).
sensor_value(130)::pursuer(2).
sensor_value(131)::pursuer(3).
sensor_value(132)::pursuer(4).
sensor_value(133)::pursuer(5).
sensor_value(134)::pursuer(6).
sensor_value(135)::pursuer(7).
sensor_value(136)::pursuer(8).
sensor_value(137)::pursuer(9).
sensor_value(138)::pursuer(10).
sensor_value(139)::pursuer(11).
sensor_value(140)::pursuer(12).
sensor_value(141)::pursuer(13).
sensor_value(142)::pursuer(14).
sensor_value(143)::pursuer(15).
sensor_value(144)::pursuer(16).
sensor_value(145)::pursuer(17).
sensor_value(146)::pursuer(18).
sensor_value(147)::pursuer(19).
sensor_value(148)::pursuer(20).
sensor_value(149)::pursuer(21).
sensor_value(150)::pursuer(22).
sensor_value(151)::pursuer(23).
sensor_value(152)::pursuer(24).
sensor_value(153)::pursuer(25).
sensor_value(154)::pursuer(26).
sensor_value(155)::pursuer(27).
sensor_value(156)::pursuer(28).
sensor_value(157)::pursuer(29).
sensor_value(158)::pursuer(30).
sensor_value(159)::pursuer(31).

% 30 sensors for collisions
sensor_value(160)::food_collision.
sensor_value(161)::poison_collision.

% transition
sensor_position(X, right) :- sensor_position(X), X > 29.               % 30, 31, 0, 1, 2
sensor_position(X, right) :- sensor_position(X), X < 3.              
sensor_position(X, upright) :- sensor_position(X), X > 1, X < 7.      % 2, 3, 4, 5, 6
sensor_position(X, up) :- sensor_position(X), X < 11, X > 5.           % 6, 7, 8, 9, 10
sensor_position(X, upleft) :- sensor_position(X), X > 9, X < 15.      % 10, 11, 12, 13, 14
sensor_position(X, left) :- sensor_position(X), X > 13, X < 19.        % 14, 15, 16, 17, 18
sensor_position(X, downleft) :- sensor_position(X), X > 17, X < 23.   % 18, 19, 20, 21, 22
sensor_position(X, down) :- sensor_position(X), X > 21, X < 27.        % 22, 23, 24, 25, 26
sensor_position(X, downright) :- sensor_position(X), X > 25, X < 31.  % 26, 27, 28, 29, 30

opposite_direction(up, down).
opposite_direction(left, right).
opposite_direction(upleft, downright).
opposite_direction(upright, downleft).
opposite_direction(X,Y) :- opposite_direction(Y,X).


% % it is unsafe to move towards poison
% unsafe_next :- action(up), poison(X), sensor_position_up(X).
% unsafe_next :- action(down), poison(X), sensor_position_down(X).
% unsafe_next :- action(left), poison(X), sensor_position_left(X).
% unsafe_next :- action(right), poison(X), sensor_position_right(X).

% it is unsafe to not move away from poison
go_away_from_poison :- poison(Id), sensor_position(Id, Loc), sensor_position(Id), action(Opp), opposite_direction(Loc, Opp).

% if there is no poison on a side, that action is safe
nopoison :- not poison(Id), sensor_position(Id, Loc), sensor_position(Id), action(Loc).

% % it is unsafe to move towards barrier
% go_away_from_barrier :- barrier(Id), sensor_position(Id, Loc), sensor_position(Id), action(Opp), opposite_direction(Loc, Opp).

% % it is unsafe to move towards barrier
% go_away_from_obstacle :- obstacle(Id), sensor_position(Id, Loc), sensor_position(Id), action(Opp), opposite_direction(Loc, Opp).

% % it is unsafe to move towards pursuer
% go_away_from_pursuer :- pursuer(Id), obstacle(Id), sensor_position(Id, Loc), sensor_position(Id), action(Opp), opposite_direction(Loc, Opp).

% % it is safe to move towards food
go_towards_food :- food(Id), sensor_position(Id, Loc), sensor_position(Id), action(Loc).

% if there is poison on both sides, it is safe to not move
% stay_put :- action(none), poison(X1), sensor_position_down(X1), sensor_position_up(X2), poison(X2), sensor_position(X1), sensor_position(X2).
% stay_put :- action(none), poison(X1), sensor_position_left(X1), sensor_position_right(X2), poison(X2), sensor_position(X1), sensor_position(X2).


% safe_next :- go_away_from_barrier.
safe_next :- go_away_from_poison.
% safe_next :- go_away_from_obstacle.
% safe_next :- go_away_from_pursuer.
safe_next :- go_towards_food.
safe_next :- nopoison.
% safe_next :- stay_put.

sensor_position(0).
sensor_position(1).
sensor_position(2).
sensor_position(3).
sensor_position(4).
sensor_position(5).
sensor_position(6).
sensor_position(7).
sensor_position(8).
sensor_position(9).
sensor_position(10).
sensor_position(11).
sensor_position(12).
sensor_position(13).
sensor_position(14).
sensor_position(15).
sensor_position(16).
sensor_position(17).
sensor_position(18).
sensor_position(19).
sensor_position(20).
sensor_position(21).
sensor_position(22).
sensor_position(23).
sensor_position(24).
sensor_position(25).
sensor_position(26).
sensor_position(27).
sensor_position(28).
sensor_position(29).
sensor_position(30).
sensor_position(31).

