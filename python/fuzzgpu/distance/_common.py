"""Shared helpers and rapidfuzz-compatible alignment data structures."""


def cutoff_distance(value, cutoff):
    return value if cutoff is None or value <= cutoff else cutoff + 1


def normalized_distance(distance, maximum):
    return 0.0 if maximum == 0 else distance / maximum


def _list_to_editops(ops, src_len, dest_len):
    if not ops:
        return []

    if len(ops[0]) == 5:
        return Opcodes(ops, src_len, dest_len).as_editops()._editops

    blocks = []
    for op in ops:
        edit_type, src_pos, dest_pos = op

        if src_pos > src_len or dest_pos > dest_len:
            raise ValueError("List of edit operations invalid")
        if src_pos == src_len and edit_type != "insert":
            raise ValueError("List of edit operations invalid")
        if dest_pos == dest_len and edit_type != "delete":
            raise ValueError("List of edit operations invalid")

        # keep operations are not relevant in editops
        if edit_type == "equal":
            continue

        blocks.append(Editop(edit_type, src_pos, dest_pos))

    # validate order of editops
    for i in range(len(blocks) - 1):
        if blocks[i + 1].src_pos < blocks[i].src_pos or blocks[i + 1].dest_pos < blocks[i].dest_pos:
            raise ValueError("List of edit operations out of order")
        if blocks[i + 1].src_pos == blocks[i].src_pos and blocks[i + 1].dest_pos == blocks[i].dest_pos:
            raise ValueError("Duplicated edit operation")

    return blocks


def _list_to_opcodes(ops, src_len, dest_len):
    if not ops or len(ops[0]) == 3:
        return Editops(ops, src_len, dest_len).as_opcodes()._opcodes

    blocks = []
    for op in ops:
        edit_type, src_start, src_end, dest_start, dest_end = op

        if src_end > src_len or dest_end > dest_len:
            raise ValueError("List of edit operations invalid")
        if src_end < src_start or dest_end < dest_start:
            raise ValueError("List of edit operations invalid")

        if edit_type in {"equal", "replace"} and (
            src_end - src_start != dest_end - dest_start or src_start == src_end
        ):
            raise ValueError("List of edit operations invalid")
        if edit_type == "insert" and (src_start != src_end or dest_start == dest_end):
            raise ValueError("List of edit operations invalid")
        if edit_type == "delete" and (src_start == src_end or dest_start != dest_end):
            raise ValueError("List of edit operations invalid")

        # merge similar adjacent blocks
        if blocks and (
            blocks[-1].tag == edit_type
            and blocks[-1].src_end == src_start
            and blocks[-1].dest_end == dest_start
        ):
            blocks[-1].src_end = src_end
            blocks[-1].dest_end = dest_end
            continue

        blocks.append(Opcode(edit_type, src_start, src_end, dest_start, dest_end))

    if blocks[0].src_start != 0 or blocks[0].dest_start != 0:
        raise ValueError("List of edit operations does not start at position 0")
    if blocks[-1].src_end != src_len or blocks[-1].dest_end != dest_len:
        raise ValueError("List of edit operations does not end at the string ends")
    for i in range(len(blocks) - 1):
        if blocks[i + 1].src_start != blocks[i].src_end or blocks[i + 1].dest_start != blocks[i].dest_end:
            raise ValueError("List of edit operations is not continuous")

    return blocks


class MatchingBlock:
    """Tuple like object describing a matching subsequence (src_start, dest_start, size)."""

    def __init__(self, src_start, dest_start, size):
        self.src_start = src_start
        self.dest_start = dest_start
        self.size = size

    def __len__(self):
        return 3

    def __eq__(self, other):
        try:
            if len(other) != 3:
                return False
            return bool(
                other[0] == self.src_start and other[1] == self.dest_start and other[2] == self.size
            )
        except TypeError:
            return False

    def __getitem__(self, i):
        if i in {0, -3}:
            return self.src_start
        if i in {1, -2}:
            return self.dest_start
        if i in {2, -1}:
            return self.size
        raise IndexError("MatchingBlock index out of range")

    def __iter__(self):
        for i in range(3):
            yield self[i]

    def __repr__(self):
        return f"MatchingBlock(src_start={self.src_start}, dest_start={self.dest_start}, size={self.size})"


class Editop:
    """Tuple like object describing an edit operation (tag, src_pos, dest_pos)."""

    def __init__(self, tag, src_pos, dest_pos):
        self.tag = tag
        self.src_pos = src_pos
        self.dest_pos = dest_pos

    def __len__(self):
        return 3

    def __eq__(self, other):
        try:
            if len(other) != 3:
                return False
            return bool(other[0] == self.tag and other[1] == self.src_pos and other[2] == self.dest_pos)
        except TypeError:
            return False

    def __getitem__(self, i):
        if i in {0, -3}:
            return self.tag
        if i in {1, -2}:
            return self.src_pos
        if i in {2, -1}:
            return self.dest_pos
        raise IndexError("Editop index out of range")

    def __iter__(self):
        for i in range(3):
            yield self[i]

    def __repr__(self):
        return f"Editop(tag={self.tag!r}, src_pos={self.src_pos}, dest_pos={self.dest_pos})"


class Editops:
    """List like object of Editops describing how to turn s1 into s2."""

    def __init__(self, editops=None, src_len=0, dest_len=0):
        self._src_len = src_len
        self._dest_len = dest_len
        self._editops = _list_to_editops(editops, src_len, dest_len)

    @classmethod
    def from_opcodes(cls, opcodes):
        return opcodes.as_editops()

    def as_opcodes(self):
        x = Opcodes.__new__(Opcodes)
        x._src_len = self._src_len
        x._dest_len = self._dest_len
        blocks = []
        src_pos = 0
        dest_pos = 0
        i = 0
        while i < len(self._editops):
            if src_pos < self._editops[i].src_pos or dest_pos < self._editops[i].dest_pos:
                blocks.append(
                    Opcode(
                        "equal",
                        src_pos,
                        self._editops[i].src_pos,
                        dest_pos,
                        self._editops[i].dest_pos,
                    )
                )
                src_pos = self._editops[i].src_pos
                dest_pos = self._editops[i].dest_pos

            src_begin = src_pos
            dest_begin = dest_pos
            tag = self._editops[i].tag
            while (
                i < len(self._editops)
                and self._editops[i].tag == tag
                and src_pos == self._editops[i].src_pos
                and dest_pos == self._editops[i].dest_pos
            ):
                if tag == "replace":
                    src_pos += 1
                    dest_pos += 1
                elif tag == "insert":
                    dest_pos += 1
                elif tag == "delete":
                    src_pos += 1
                i += 1

            blocks.append(Opcode(tag, src_begin, src_pos, dest_begin, dest_pos))

        if src_pos < self.src_len or dest_pos < self.dest_len:
            blocks.append(Opcode("equal", src_pos, self.src_len, dest_pos, self.dest_len))

        x._opcodes = blocks
        return x

    def as_matching_blocks(self):
        blocks = []
        src_pos = 0
        dest_pos = 0
        for op in self:
            if src_pos < op.src_pos or dest_pos < op.dest_pos:
                length = min(op.src_pos - src_pos, op.dest_pos - dest_pos)
                if length > 0:
                    blocks.append(MatchingBlock(src_pos, dest_pos, length))
                src_pos = op.src_pos
                dest_pos = op.dest_pos

            if op.tag == "replace":
                src_pos += 1
                dest_pos += 1
            elif op.tag == "delete":
                src_pos += 1
            elif op.tag == "insert":
                dest_pos += 1

        if src_pos < self.src_len or dest_pos < self.dest_len:
            length = min(self.src_len - src_pos, self.dest_len - dest_pos)
            if length > 0:
                blocks.append(MatchingBlock(src_pos, dest_pos, length))

        blocks.append(MatchingBlock(self.src_len, self.dest_len, 0))
        return blocks

    def as_list(self):
        return [tuple(op) for op in self._editops]

    def copy(self):
        x = Editops.__new__(Editops)
        x._src_len = self._src_len
        x._dest_len = self._dest_len
        x._editops = self._editops[::]
        return x

    def inverse(self):
        blocks = []
        for op in self:
            tag = op.tag
            if tag == "delete":
                tag = "insert"
            elif tag == "insert":
                tag = "delete"
            blocks.append(Editop(tag, op.dest_pos, op.src_pos))

        x = Editops.__new__(Editops)
        x._src_len = self.dest_len
        x._dest_len = self.src_len
        x._editops = blocks
        return x

    def remove_subsequence(self, subsequence):
        result = Editops.__new__(Editops)
        result._src_len = self._src_len
        result._dest_len = self._dest_len

        if len(subsequence) > len(self):
            raise ValueError("subsequence is not a subsequence")

        result._editops = [None] * (len(self) - len(subsequence))

        offset = 0
        op_pos = 0
        result_pos = 0

        for sop in subsequence:
            while op_pos != len(self) and sop != self._editops[op_pos]:
                result._editops[result_pos] = self._editops[op_pos]
                result._editops[result_pos].src_pos += offset
                result_pos += 1
                op_pos += 1

            if op_pos == len(self):
                raise ValueError("subsequence is not a subsequence")

            if sop.tag == "insert":
                offset += 1
            elif sop.tag == "delete":
                offset -= 1

            op_pos += 1

        while op_pos != len(self):
            result._editops[result_pos] = self._editops[op_pos]
            result._editops[result_pos].src_pos += offset
            result_pos += 1
            op_pos += 1

        return result

    def apply(self, source_string, destination_string):
        res_str = ""
        src_pos = 0

        for op in self._editops:
            while src_pos < op.src_pos:
                res_str += source_string[src_pos]
                src_pos += 1

            if op.tag == "replace":
                res_str += destination_string[op.dest_pos]
                src_pos += 1
            elif op.tag == "insert":
                res_str += destination_string[op.dest_pos]
            elif op.tag == "delete":
                src_pos += 1

        while src_pos < len(source_string):
            res_str += source_string[src_pos]
            src_pos += 1

        return res_str

    @property
    def src_len(self):
        return self._src_len

    @src_len.setter
    def src_len(self, value):
        self._src_len = value

    @property
    def dest_len(self):
        return self._dest_len

    @dest_len.setter
    def dest_len(self, value):
        self._dest_len = value

    def __eq__(self, other):
        if not isinstance(other, Editops):
            return False
        return (
            self.dest_len == other.dest_len
            and self.src_len == other.src_len
            and self._editops == other._editops
        )

    def __len__(self):
        return len(self._editops)

    def __delitem__(self, key):
        del self._editops[key]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._editops[key]

        start, stop, step = key.indices(len(self._editops))
        if step < 0:
            raise ValueError("step sizes below 0 lead to an invalid order of editops")

        x = Editops.__new__(Editops)
        x._src_len = self._src_len
        x._dest_len = self._dest_len
        x._editops = self._editops[start:stop:step]
        return x

    def __iter__(self):
        yield from self._editops

    def __repr__(self):
        return (
            "Editops(["
            + ", ".join(repr(op) for op in self)
            + f"], src_len={self.src_len}, dest_len={self.dest_len})"
        )


class Opcode:
    """Tuple like object describing an edit operation
    (tag, src_start, src_end, dest_start, dest_end)."""

    def __init__(self, tag, src_start, src_end, dest_start, dest_end):
        self.tag = tag
        self.src_start = src_start
        self.src_end = src_end
        self.dest_start = dest_start
        self.dest_end = dest_end

    def __len__(self):
        return 5

    def __eq__(self, other):
        try:
            if len(other) != 5:
                return False
            return bool(
                other[0] == self.tag
                and other[1] == self.src_start
                and other[2] == self.src_end
                and other[3] == self.dest_start
                and other[4] == self.dest_end
            )
        except TypeError:
            return False

    def __getitem__(self, i):
        if i in {0, -5}:
            return self.tag
        if i in {1, -4}:
            return self.src_start
        if i in {2, -3}:
            return self.src_end
        if i in {3, -2}:
            return self.dest_start
        if i in {4, -1}:
            return self.dest_end
        raise IndexError("Opcode index out of range")

    def __iter__(self):
        for i in range(5):
            yield self[i]

    def __repr__(self):
        return (
            f"Opcode(tag={self.tag!r}, src_start={self.src_start}, src_end={self.src_end}, "
            f"dest_start={self.dest_start}, dest_end={self.dest_end})"
        )


class Opcodes:
    """List like object of Opcodes describing how to turn s1 into s2."""

    def __init__(self, opcodes=None, src_len=0, dest_len=0):
        self._src_len = src_len
        self._dest_len = dest_len
        self._opcodes = _list_to_opcodes(opcodes, src_len, dest_len)

    @classmethod
    def from_editops(cls, editops):
        return editops.as_opcodes()

    def as_editops(self):
        x = Editops.__new__(Editops)
        x._src_len = self._src_len
        x._dest_len = self._dest_len
        blocks = []
        for op in self:
            if op.tag == "replace":
                for j in range(op.src_end - op.src_start):
                    blocks.append(Editop("replace", op.src_start + j, op.dest_start + j))
            elif op.tag == "insert":
                for j in range(op.dest_end - op.dest_start):
                    blocks.append(Editop("insert", op.src_start, op.dest_start + j))
            elif op.tag == "delete":
                for j in range(op.src_end - op.src_start):
                    blocks.append(Editop("delete", op.src_start + j, op.dest_start))

        x._editops = blocks
        return x

    def as_matching_blocks(self):
        blocks = []
        for op in self:
            if op.tag == "equal":
                length = min(op.src_end - op.src_start, op.dest_end - op.dest_start)
                if length > 0:
                    blocks.append(MatchingBlock(op.src_start, op.dest_start, length))

        blocks.append(MatchingBlock(self.src_len, self.dest_len, 0))
        return blocks

    def as_list(self):
        return [tuple(op) for op in self._opcodes]

    def copy(self):
        x = Opcodes.__new__(Opcodes)
        x._src_len = self._src_len
        x._dest_len = self._dest_len
        x._opcodes = self._opcodes[::]
        return x

    def inverse(self):
        blocks = []
        for op in self:
            tag = op.tag
            if tag == "delete":
                tag = "insert"
            elif tag == "insert":
                tag = "delete"
            blocks.append(Opcode(tag, op.dest_start, op.dest_end, op.src_start, op.src_end))

        x = Opcodes.__new__(Opcodes)
        x._src_len = self.dest_len
        x._dest_len = self.src_len
        x._opcodes = blocks
        return x

    def apply(self, source_string, destination_string):
        res_str = ""
        for op in self._opcodes:
            if op.tag == "equal":
                res_str += source_string[op.src_start : op.src_end]
            elif op.tag in {"replace", "insert"}:
                res_str += destination_string[op.dest_start : op.dest_end]
        return res_str

    @property
    def src_len(self):
        return self._src_len

    @src_len.setter
    def src_len(self, value):
        self._src_len = value

    @property
    def dest_len(self):
        return self._dest_len

    @dest_len.setter
    def dest_len(self, value):
        self._dest_len = value

    def __eq__(self, other):
        if not isinstance(other, Opcodes):
            return False
        return (
            self.dest_len == other.dest_len
            and self.src_len == other.src_len
            and self._opcodes == other._opcodes
        )

    def __len__(self):
        return len(self._opcodes)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._opcodes[key]
        raise TypeError("Expected index")

    def __iter__(self):
        yield from self._opcodes

    def __repr__(self):
        return (
            "Opcodes(["
            + ", ".join(repr(op) for op in self)
            + f"], src_len={self.src_len}, dest_len={self.dest_len})"
        )


class ScoreAlignment:
    """Tuple like object describing the position of the compared strings in
    src and dest (score, src_start, src_end, dest_start, dest_end)."""

    def __init__(self, score, src_start, src_end, dest_start, dest_end):
        self.score = score
        self.src_start = src_start
        self.src_end = src_end
        self.dest_start = dest_start
        self.dest_end = dest_end

    def __len__(self):
        return 5

    def __eq__(self, other):
        try:
            if len(other) != 5:
                return False
            return bool(
                other[0] == self.score
                and other[1] == self.src_start
                and other[2] == self.src_end
                and other[3] == self.dest_start
                and other[4] == self.dest_end
            )
        except TypeError:
            return False

    def __getitem__(self, i):
        if i in {0, -5}:
            return self.score
        if i in {1, -4}:
            return self.src_start
        if i in {2, -3}:
            return self.src_end
        if i in {3, -2}:
            return self.dest_start
        if i in {4, -1}:
            return self.dest_end
        raise IndexError("ScoreAlignment index out of range")

    def __iter__(self):
        for i in range(5):
            yield self[i]

    def __repr__(self):
        return (
            f"ScoreAlignment(score={self.score}, src_start={self.src_start}, "
            f"src_end={self.src_end}, dest_start={self.dest_start}, dest_end={self.dest_end})"
        )
