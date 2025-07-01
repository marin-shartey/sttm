# Wraps a sparse document-term matrix for text (bag-of-words).
# Exposes `.feature_names` (vocabulary list) and allows slicing by message IDs.
# `positions` is a pandas Series mapping message_id to row index in the matrix.
class TextFeatures:
    def __init__(self, matrix, positions, feature_names):
        """
        matrix: scipy sparse matrix of shape (n_messages, n_features)
        positions: pd.Series index=message_id, value=row index in matrix
        feature_names: list or array of token strings
        """
        self.matrix = matrix
        self.positions = positions
        self.feature_names = feature_names

    def __getitem__(self, msg_ids):
        """
        Returns the submatrix for the given message IDs (rows).
        msg_ids: array-like of message_id values
        """
        # Get row indices for these message IDs (drop IDs not present)
        row_inds = self.positions.reindex(msg_ids).dropna().astype(int).values
        return self.matrix[row_inds]
