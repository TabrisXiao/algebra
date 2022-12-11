// RUN: mc-opt %s -canonicalize | FileCheck %s

func.func @invers_canonicalize (%arg0: !AA.AElement<Symbol: @Abel>) -> (!AA.AElement<Symbol: @Abel>){
    // CHECK-NEXT: return %[[RESULT:.*]] : !AA.AElement<Symbol: @Abel>

    %0 = AA.Inverse (%arg0) : !AA.AElement<Symbol: @Abel> -> !AA.AElement<Symbol: @Abel>
    %1 = AA.Inverse (%0) : !AA.AElement<Symbol: @Abel> -> !AA.AElement<Symbol: @Abel>
    func.return %1 : !AA.AElement<Symbol: @Abel>
}
