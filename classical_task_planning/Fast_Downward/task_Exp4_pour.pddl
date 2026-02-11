;; problem file: task_Exp4_pour.pddl
;; This problem file is to run Fast Downward for the "pour" task. It is for Experiment 4.

(define (problem bimanual-probPour)
  (:domain domain_pour)
(:objects mug sugar_box pitcher_base bowl tuna_fish_can tomato_soup_can pudding_box mustard_bottle gelatin_box cracker_box initial_location_handleft initial_location_handright - graspable 
            handright handleft - hand
            1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 - location 
)

  (:init 
  	(at handleft initial_location_handleft)
    (at handright initial_location_handright)
  )
 (:goal (and (poured gelatin_box 23 bowl 11)))) 