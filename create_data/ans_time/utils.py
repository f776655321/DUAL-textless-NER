import torch

from dataclasses import dataclass

import torchaudio
import random



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

def merge_repeats(path,transcript):
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

def force_align(model,audio_dir,SPEECH_FILE,dictionary,text):
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #extract wave feature
    with torch.inference_mode():
        waveform, _ = torchaudio.load( audio_dir + SPEECH_FILE + '.ogg')
        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()
    #tokenize transcript
    tokens = [dictionary[c] for c in text]

    trellis = get_trellis(emission, tokens)

    #determine where the tokens come from
    path = backtrack(trellis, emission, tokens)

    segments = merge_repeats(path,text)

    word_segments = merge_words(segments)

    return word_segments,waveform.size(1),trellis.size(0)

def get_ans_time(target, word_segments,waveform_size,trellis_size,find_time,sample_rate):
    start_index = 0
    start_time = 0

    end_index = len(target) - 1
    end_time = 0

    find_target = False

    not_find = True

    start_finded = False
    
    x0 = 0
    x1 = 0

    word_start_index = 0
    word_end_index = 0

    for word in word_segments:
        
        if(target[start_index] == word.label or target[start_index].replace("'s",'') == word.label.replace("'s",'') or target[start_index].replace("'",'') == word.label.replace("'",'') or target[start_index].replace(".",'') == word.label.replace(".",'')):
            ratio = waveform_size / (trellis_size - 1)
            x0 = int(ratio * word.start)
            word_start_index = word.start
            start_finded = True

        if(target[end_index] == word.label or target[end_index].replace("'s",'') == word.label.replace("'s",'') or target[end_index].replace("'",'') == word.label.replace("'",'') or target[end_index].replace(".",'') == word.label.replace(".",'')):
            ratio = waveform_size / (trellis_size - 1)
            x1 = int(ratio * word.end)
            word_end_index = word.end
            
            if(start_finded == True):
                find_target = True

        if(find_target == True and (word_start_index,word_end_index) not in find_time):
            
            start_time = x0 / sample_rate
            end_time = x1 / sample_rate
            # start.append(round(start_time,3))

            # start.append(word_start_index)
            # end.append(round(end_time,3))
            
            
            # end.append(word_end_index)
            find_time.add((word_start_index,word_end_index))
            not_find = False
            start_finded = False
            find_target = False
            break

        elif(find_target == True and (word_start_index,word_end_index) in find_time):
            find_target = False
            start_finded = False

    if(not_find == True):
        start_time = 0
        end_time = 0
    
    return start_time,end_time

def random_choice_without_repeat(array, num_choices):
    shuffled = array.copy()
    random.shuffle(shuffled)
    choices = []
    while len(choices) < num_choices and shuffled:
        choices.append(shuffled.pop())
    return choices








