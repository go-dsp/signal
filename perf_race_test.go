//go:build race

package signal

// The race detector adds allocations: refbench's allocation check is only
// exact in a plain build.
const perfRace = true
