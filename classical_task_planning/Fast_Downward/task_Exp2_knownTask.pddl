;; problem file: task_bimanual_3.pddl

(define (problem bimanual-prob1)
  (:domain domain_pour)
(:objects cup pitcher initial_location_handleft initial_location_handright - graspable 
            handright handleft - hand
            31 11 - location
  )

  (:init 
  	(at handleft initial_location_handleft)
    (at handright initial_location_handright)
  )

 (:goal (and (poured pitcher 31 cup 11)))) 