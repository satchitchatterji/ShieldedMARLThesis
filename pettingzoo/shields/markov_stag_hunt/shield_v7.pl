% create a shield that goes towards the stag and waits until there is another before hunting

% actions
action(0)::action(left);
action(1)::action(right);
action(2)::action(up);
action(3)::action(down);
action(4)::action(stay).

sensor_value(0)::sensor(left).
sensor_value(1)::sensor(right).
sensor_value(2)::sensor(up).
sensor_value(3)::sensor(down).
sensor_value(4)::sensor(stag_near_self).
sensor_value(5)::sensor(stag_near_other).

% define movement towards the stag
go_towards_stag :- action(Dir), sensor(Dir).

% it is unsafe to not go towards the stag if it not near
unsafe_next :- \+go_towards_stag, \+sensor(stag_near_self).

% it is unsafe to not stay in place if the stag is near and there is no other agent
unsafe_next :-  sensor(stag_near_self), \+sensor(stag_near_other), \+action(stay).

% it is unsafe to not hunt if the stag is near and there is another agent
unsafe_next :-  sensor(stag_near_self), sensor(stag_near_other), \+go_towards_stag.

safe_next :- \+unsafe_next.

% if stag is not near self
%   then you must move towards the stag
% else if stag is near self
%   if stag is not near other
%     then you must stay
%   else
%     then you must move towards the stag