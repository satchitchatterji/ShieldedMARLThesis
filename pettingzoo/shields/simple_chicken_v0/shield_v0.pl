% actions
action(0)::action(swerve);
action(1)::action(straight).

sensor_value(0)::sensor(swerve).
sensor_value(1)::sensor(straight).

unsafe_next :- action(straight).
safe_next:- \+unsafe_next.
