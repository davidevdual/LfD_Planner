;; problem file: task_bimanual_1.pddl

(define (problem bimanual-prob1)
  (:domain domain_pour)
  (:objects cup pitcher initial_location_handleft initial_location_handright - graspable
            handright handleft - hand
            1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 - location
  )

  (:init 
  	(at handleft initial_location_handleft)
    (at handright initial_location_handright)
  )

 (:goal (and (poured pitcher 15 cup 31)))) 