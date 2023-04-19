#reference from https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

import torch
import torchaudio
from dataclasses import dataclass

import IPython
import matplotlib
import matplotlib.pyplot as plt


#transcript need to be in the form like : I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT
def alignment(audio_file,transcript,target):
    print(torch.__version__)
    print(torchaudio.__version__)
    matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

    torch.random.manual_seed(0)

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    SPEECH_FILE = audio_file

    #extract wave feature
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    print(labels)
    with torch.inference_mode():
        waveform, _ = torchaudio.load(SPEECH_FILE)
        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()

    #tokenize transcript
    dictionary = {c.lower(): i for i, c in enumerate(labels)}
    tokens = [dictionary[c] for c in transcript]
    #get probability
    trellis = get_trellis(emission, tokens)

    #determine where the word come from
    path = backtrack(trellis, emission, tokens)

    segments = merge_repeats(path)

    word_segments = merge_words(segments)
    start_index = 0
    start_time = 0

    end_index = len(target) - 1
    end_time = 0
    for word in word_segments:
        if(word.label == target[start_index]):
            print(word.label)
            print(target[start_index])
            ratio = waveform.size(1) / (trellis.size(0) - 1)
            x0 = int(ratio * word.start)
            start_time = x0 / bundle.sample_rate
        
        if(word.label == target[end_index]):
            print(word.label)
            print(target[end_index])
            ratio = waveform.size(1) / (trellis.size(0) - 1)
            x0 = int(ratio * word.end)
            end_time = x0 / bundle.sample_rate

    return (round(start_time,3),round(end_time,3))

#From the emission matrix, next we generate the trellis which represents the probability of transcript labels occur at each time frame.
def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

#Once the trellis is generated, we will traverse it following the elements with high probability.
def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

SPEECH_FILE = '20110119-0900-PLENARY-3-en_20110119-09:05:29_2.ogg'

data_dir = '/tmp2/b09902111/slue-voxpopuli/fine-tune/'

transcript = 'this could be done by looking more specifically into the suggested fallback clause in article twenty one in cases brought by the employee against the employer defining as relevant the place of business from which the employee receives or received day to day instructions'

old_t = transcript.replace(' ', '|')
transcript = transcript.replace(' ', '|').replace('.','')

print(transcript)

a = 86
print(old_t[86:86 + 18])
# time = alignment(data_dir + SPEECH_FILE,transcript,target)

# print(time)

