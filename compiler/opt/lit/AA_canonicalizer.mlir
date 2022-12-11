
func.func @main () -> (AElement <Symbol: @Abel>){
    %cst0 = AA.AElemDecl {encoding = @Abel} -> AElement <Symbol: @Abel>
    %0 = AA.Inverse (%cst0) : AElement <Symbol: @Abel> -> AElement <Symbol: @Abel>
    return %0 : AElement <Symbol: @Abel>
}
