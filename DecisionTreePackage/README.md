This is a Decision Tree library developed by Divyanshu Tripathy for
DS 4350 in the University of Utah.

The tree is called by creating an instance with some data.
We can build the tree using the build command. We need to pass in the splitting criteria (the function for ginni, majority error, entropy).
This can be done by passing a delegate such as decision_tree.ginni_index, decision_tree.majority_error, decision_tree.entropy or a function chosen by the user.