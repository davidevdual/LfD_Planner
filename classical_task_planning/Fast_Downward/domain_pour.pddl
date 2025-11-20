;; domain_1.pddl for bimanual manipulation
;; This domain is for the pouring goal exclusively

(define (domain domain_pour)
  (:requirements :strips)

  (:types graspable - object
               hand - object
           location - object
  )

  (:predicates 
  	(at ?hand - hand ?obj - graspable)
    (enclosed ?hand - hand ?obj - graspable)
    (poured ?fromObj - graspable ?fromLoc - location ?toObj - graspable ?toLoc - location)
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

   (:action pour
           :parameters (?hand1 ?hand2 - hand ?fromObj - graspable ?fromLoc - location ?toObj - graspable ?toLoc - location)
           :precondition (and (at ?hand1 ?toObj) (enclosed ?hand1 ?fromObj) (enclosed ?hand2 ?toObj) (not (enclosed ?hand1 ?toObj)) (not (enclosed ?hand2 ?fromObj)))
           :effect (and (poured ?fromObj ?fromLoc ?toObj ?toLoc))
   )
)