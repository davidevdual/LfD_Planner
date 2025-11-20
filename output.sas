begin_version
3
end_version
begin_metric
0
end_metric
9
begin_variable
var0
-1
2
Atom at(handright, pitcher)
NegatedAtom at(handright, pitcher)
end_variable
begin_variable
var1
-1
2
Atom enclosed(handright, pitcher)
NegatedAtom enclosed(handright, pitcher)
end_variable
begin_variable
var2
-1
2
Atom at(handright, cup)
NegatedAtom at(handright, cup)
end_variable
begin_variable
var3
-1
2
Atom enclosed(handright, cup)
NegatedAtom enclosed(handright, cup)
end_variable
begin_variable
var4
-1
2
Atom at(handleft, pitcher)
NegatedAtom at(handleft, pitcher)
end_variable
begin_variable
var5
-1
2
Atom enclosed(handleft, pitcher)
NegatedAtom enclosed(handleft, pitcher)
end_variable
begin_variable
var6
-1
2
Atom at(handleft, cup)
NegatedAtom at(handleft, cup)
end_variable
begin_variable
var7
-1
2
Atom enclosed(handleft, cup)
NegatedAtom enclosed(handleft, cup)
end_variable
begin_variable
var8
-1
2
Atom poured(pitcher, 15, cup, 20)
NegatedAtom poured(pitcher, 15, cup, 20)
end_variable
0
begin_state
1
1
1
1
1
1
1
1
1
end_state
begin_goal
1
8 0
end_goal
10
begin_operator
approach handleft cup
0
1
0 6 1 0
1
end_operator
begin_operator
approach handleft pitcher
0
1
0 4 1 0
1
end_operator
begin_operator
approach handright cup
0
1
0 2 1 0
1
end_operator
begin_operator
approach handright pitcher
0
1
0 0 1 0
1
end_operator
begin_operator
enclose handleft cup
1
6 0
1
0 7 1 0
1
end_operator
begin_operator
enclose handleft pitcher
1
4 0
1
0 5 1 0
1
end_operator
begin_operator
enclose handright cup
1
2 0
1
0 3 1 0
1
end_operator
begin_operator
enclose handright pitcher
1
0 0
1
0 1 1 0
1
end_operator
begin_operator
pour handleft handright pitcher 15 cup 20
5
6 0
7 1
5 0
3 0
1 1
1
0 8 -1 0
1
end_operator
begin_operator
pour handright handleft pitcher 15 cup 20
5
2 0
7 0
5 1
3 1
1 0
1
0 8 -1 0
1
end_operator
0
