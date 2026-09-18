/**
 *  What a loser of a lock does between two attempts, as `-Dwaiting=`: the policy `lock` and
 *  `lock_shared` of `spin_shared_mutex.pml`, and the slot lock of `atomic_hash_table.pml`, plug
 *  into their retry loops. A policy decides how long a waiter takes to notice that the word
 *  moved, never what the word may say, so every safety property either model asserts has to
 *  hold under all four of them, and a swap is latency rather than protocol.
 *
 *  - `spinning`, the default: the word re-read until it admits the attempt.
 *  - `pausing`: the same, with a step that touches no location between two reads.
 *  - `on_the_address`: a wait on the word itself, which returns as soon as the word differs from
 *    the one the failed attempt read - possibly one that still refuses, so the loop runs again.
 *  - `parking`: the waiter hands the core over and retries whenever it is scheduled again, which
 *    may be at any moment; it blocks on nothing and so admits the most interleavings of the four.
 *
 *  Include after `weak_memory.pml`, whose `newest_value` the address wait reads, and before the
 *  model that retries. Every policy uses the caller's `seen` scratch, which holds the word the
 *  failed attempt read.
 */

// The knob's values are integers, so a typo fails the range check below.
#define spinning 1
#define pausing 2
#define on_the_address 3
#define parking 4
#ifndef waiting
#define waiting spinning
#endif
#if waiting < spinning || waiting > parking
#error "waiting is spinning, pausing, on_the_address or parking"
#endif

// wait_until: `word` is the location waited on, `admits` what a re-reading waiter blocks on.
#if waiting == spinning
inline wait_until(word, admits) { (admits) }
#elif waiting == pausing
inline wait_until(word, admits) { skip; (admits) }
#elif waiting == on_the_address
inline wait_until(word, admits) { (newest_value(word) != seen) }
#else
inline wait_until(word, admits) { skip }
#endif
