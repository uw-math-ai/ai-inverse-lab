import MathAILab.Spec

namespace MathAILab

theorem recoveryFormula : RecoveryGoal := by
  intro K inst V B h rate q observed hh data x hx
  have hxdata := data x hx
  simp only [recoverPotential, oneStepReturn] at *
  grind

theorem fullObservationUniqueness : FullObservationGoal := by
  intro K inst V h rate q₁ q₂ hh data
  funext x
  have hxdata := data x
  simp only [oneStepReturn] at hxdata
  grind

end MathAILab

#print axioms MathAILab.recoveryFormula
#print axioms MathAILab.fullObservationUniqueness
