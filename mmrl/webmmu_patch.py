import re

def apply_webmmu_patch(source, patch):
    """Apply all signed hunks atomically using unique full-content anchors.

    WebMMU reference diffs can use stale line counts and snippets. Exact old
    content is authoritative (indentation only is normalized). Pure insertions
    require an explicit unified hunk location. No fragment-preview fallback.
    """
    fence=re.fullmatch(r'\s*```html\s*\n([\s\S]*?)\n```\s*', source)
    # A signed context/deletion line can refer to the source's literal wrapper.
    # Preserve it for matching; document composition removes the wrapper later.
    references_wrapper = any(
        line[:1] in {" ", "-"}
        and re.fullmatch(r"```(?:html)?", line[1:].strip())
        for line in patch.splitlines()
    )
    # Hunk positions refer to the source supplied to the agent, including fences.
    # Strip presentation wrapping only after applying every hunk.
    lines=source.splitlines(); groups=[]; old=[]; new=[]; position=None
    headers=0
    def flush():
        nonlocal old,new
        if old or new: groups.append((old,new,position))
        old=[];new=[]
    for line in patch.splitlines():
        match=re.match(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@',line)
        if match:
            flush(); position=int(match[1])-(0 if match[2]=='0' else 1)
        elif line.startswith('--- '):
            headers+=1
            if headers>1: raise ValueError('Multiple-file patches are unsupported')
        elif line.startswith(('+++ ','diff --git ','index ')) or line==r'\ No newline at end of file':
            continue
        elif line.startswith('-'):old.append(line[1:])
        elif line.startswith('+'):new.append(line[1:])
        elif line.startswith(' '):old.append(line[1:]);new.append(line[1:])
        else: raise ValueError('Invalid signed patch line')
    flush()
    if not groups: raise ValueError('Empty patch')
    offset=0; cursor=0
    for old,new,position in groups:
        if old:
            target=[s.strip() for s in old]
            matches=[i for i in range(cursor,len(lines)-len(old)+1)
                     if [s.strip() for s in lines[i:i+len(old)]]==target]
            declared=None if position is None else position+offset
            if declared in matches:
                start=declared
            elif len(matches)==1:
                start=matches[0]
            else:raise ValueError('Patch old content must match uniquely or at its declared location; matches='+str(len(matches)))
        else:
            if position is None: raise ValueError('Unanchored insertion')
            start=position+offset
            if not cursor<=start<=len(lines):raise ValueError('Insertion is out of range')
        lines[start:start+len(old)]=new
        cursor=start+len(new); offset+=len(new)-len(old)
    result = '\n'.join(lines)
    if fence and not references_wrapper:
        wrapped = re.fullmatch(r'\s*```html\s*\n([\s\S]*?)\n```\s*', result)
        if wrapped:
            result = wrapped[1]
    return result
