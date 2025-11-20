;; domain_2.pddl for bimanual manipulation
;; This domain is for the opening goal exclusively

(define (domain domain_open)
  (:requirements :strips)

  (:types graspable - object
               hand - object
           location - object
  )

  (:predicates 
  	(at ?hand - hand ?obj - graspable)
    (enclosed ?hand - hand ?obj - graspable)
    (opened ?obj - graspable ?objLoc - location)
  )

  (:action approach
           :parameters (?hand - hand ?obj - graspable)
           :precondition (and (not (at ?hand ?obj)))
           :effect (and (at ?hand ?obj))
   )

   (:action enclose
           :parameters (?hand - hand ?obj - graspable)
           :precondition (and (at ?hand ?obj) (not (enclosed ?hand ?obj)))
           :effect (and (enclosed ?hand ?obj))
   )

   (:action open
           :parameters (?hand1 ?hand2 - hand ?obj - graspable ?objLoc - location)
           :precondition (and (enclosed ?hand1 ?obj) (enclosed ?hand2 ?obj) (not (= ?hand1 ?hand2)))
           :effect (and (opened ?obj ?objLoc))
   )
)