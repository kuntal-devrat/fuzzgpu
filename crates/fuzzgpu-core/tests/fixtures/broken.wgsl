// Deliberately broken WGSL fixture for ShaderError coverage.
// Contains both a syntax error (empty initializer below) and a semantic
// error (string literal assigned to a u32) — either one fails naga parsing.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var count: u32 = ;
    let label: u32 = "not a number";
    let sum: u32 = gid.x + count + label;
}
