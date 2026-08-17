Would we be able to access the ranges?
Yes, absolutely, if we decide to dump them properly.

The reporting path is already set up to export:

per-state policy snapshots
action labels
legal actions
street / pot context
history / action trace
policy profile summary
So the dump can be shaped as:

state-level JSON
summarized range buckets
aggregated by infoset family
final average-policy export
In practice, the easiest data product is:

snapshot average policy at meaningful states
bucket by canonical infoset signature
aggregate action probabilities within each bucket
write JSON or CSV for later analysis
That gives you:

“player 1 acting first in 3bb pot, legal actions [check/call, fold, bet/raise]”
final action distribution
and then aggregate across all similar states to produce a range