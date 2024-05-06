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


% 32 sensors for barriers
sensor_value(8)::barrier(0).
sensor_value(9)::barrier(1).
sensor_value(10)::barrier(2).
sensor_value(11)::barrier(3).
sensor_value(12)::barrier(4).
sensor_value(13)::barrier(5).
sensor_value(14)::barrier(6).
sensor_value(15)::barrier(7).


% 32 sensors for food
sensor_value(16)::food(0).
sensor_value(17)::food(1).
sensor_value(18)::food(2).
sensor_value(19)::food(3).
sensor_value(20)::food(4).
sensor_value(21)::food(5).
sensor_value(22)::food(6).
sensor_value(23)::food(7).


% 32 sensors for poison
sensor_value(24)::poison(0).
sensor_value(25)::poison(1).
sensor_value(26)::poison(2).
sensor_value(27)::poison(3).
sensor_value(28)::poison(4).
sensor_value(29)::poison(5).
sensor_value(30)::poison(6).
sensor_value(31)::poison(7).


% 32 sensors for pursuer
sensor_value(32)::pursuer(0).
sensor_value(33)::pursuer(1).
sensor_value(34)::pursuer(2).
sensor_value(35)::pursuer(3).
sensor_value(36)::pursuer(4).
sensor_value(37)::pursuer(5).
sensor_value(38)::pursuer(6).
sensor_value(39)::pursuer(7).

% 30 sensors for collisions
sensor_value(40)::food_collision.
sensor_value(41)::poison_collision.

% transition
sensor_position(0, right).
sensor_position(1, upright).
sensor_position(2, up).
sensor_position(3, upleft).
sensor_position(4, left).
sensor_position(5, downleft).
sensor_position(6, down).
sensor_position(7, downright).

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
go_away_from_barrier :- barrier(Id), sensor_position(Id, Loc), sensor_position(Id), action(Opp), opposite_direction(Loc, Opp).

% % it is unsafe to move towards barrier
go_away_from_obstacle :- obstacle(Id), sensor_position(Id, Loc), sensor_position(Id), action(Opp), opposite_direction(Loc, Opp).

% % it is unsafe to move towards pursuer
go_away_from_pursuer :- pursuer(Id), obstacle(Id), sensor_position(Id, Loc), sensor_position(Id), action(Opp), opposite_direction(Loc, Opp).

% % it is safe to move towards food
go_towards_food :- food(Id), sensor_position(Id, Loc), sensor_position(Id), action(Loc).

% if there is poison on both sides, it is safe to not move
% stay_put :- action(none), poison(X1), sensor_position_down(X1), sensor_position_up(X2), poison(X2), sensor_position(X1), sensor_position(X2).
% stay_put :- action(none), poison(X1), sensor_position_left(X1), sensor_position_right(X2), poison(X2), sensor_position(X1), sensor_position(X2).


% safe_next :- go_away_from_barrier.
safe_next :- go_away_from_poison.
safe_next :- go_away_from_obstacle.
safe_next :- go_away_from_pursuer.
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

