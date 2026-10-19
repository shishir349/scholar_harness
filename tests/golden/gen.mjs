// Regenerates tests/golden/cases.json from acdyon-scholar's REAL prompt builders (read-only import).
// Run manually only when Scholar's prompts change:
//   node tests/golden/gen.mjs "<path to acdyon-scholar/backend/src>"
// Default path assumes the sibling checkout.
import { writeFileSync } from 'node:fs'
import { pathToFileURL, fileURLToPath } from 'node:url'
import path from 'node:path'

const here = path.dirname(fileURLToPath(import.meta.url))
const src = path.resolve(process.argv[2] || path.join(here, '../../../acdyon io/acdyon-scholar/backend/src'))
const load = (rel) => import(pathToFileURL(path.join(src, rel)).href)

const prompts = await load('prompts/chapter.prompt.js')
const prof = await load('utils/sectionProfiles.js')

const thesis = {
  title: 'Digital Transformation and Firm Performance in Indian SMEs',
  university: 'Example University', programme: 'PhD Management',
  methodology: 'Mixed methods', domain: 'Business and Management', geographic_focus: 'India',
}
const thesisNoGeo = { ...thesis, geographic_focus: '' }

const refs = [
  'Bandura, A. (1977). Self-efficacy: Toward a unifying theory of behavioral change. Psychological Review, 84(2), 191–215.',
  'Senge, P. M. (1990). The fifth discipline. Doubleday.',
]

const chunk1 = 'Digital adoption in small firms is uneven. Smith (2019) reports gains; Jones (2020) does not.\n\nThe Literature rarely addresses Capability. Capability shapes Adoption, and Adoption shapes Capability again.'
const chunk2 = 'What matters here is scale. Scale changes Capability in ways the earlier work ignores.'

const cases = [
  ['Chapter 1: Introduction', 1, thesis, refs, true],
  ['Chapter 2: Literature Review', 2, thesis, refs, true],
  ['Chapter 3: Research Methodology', 3, thesisNoGeo, [], false],
  ['Chapter 4: Findings and Results', 4, thesis, refs, true],
  ['Chapter 5: Discussion', 5, thesis, refs, false],
  ['Chapter 6: Conclusion and Recommendations', 6, thesisNoGeo, refs, true],
].map(([title, number, th, references, withSubs], i) => {
  const chapter = { chapter_number: number, title }
  const section = {
    number: `${number}.${i + 1}`, title: 'Section under test',
    ...(withSubs ? { subsections: [{ number: `${number}.${i + 1}.1`, title: 'Sub A' }, { number: `${number}.${i + 1}.2`, title: 'Sub B' }] } : {}),
  }
  const profile = prof.detectProfile(title, number)
  const wordsTarget = 450
  const usedConcepts = prompts.extractUsedConcepts(chunk1 + ' ' + chunk2)
  const b = {
    profileBlock: prof.buildProfileBlock(profile), noiseBlock: prof.buildNoiseBlock(profile),
    empiricalBlock: prof.buildEmpiricalBlock(profile),
  }
  return {
    name: title, thesis: th, chapter, section, references, wordsTarget, chunk1, chunk2,
    profileId: profile.id, usedConcepts,
    call1: prompts.buildSectionCall1Prompt({ thesis: th, chapter, section, wordsTarget, references, ...b, visualDirective: {} }),
    call2: prompts.buildSectionCall2Prompt({ thesis: th, chapter, section, chunk1, usedConcepts, wordsTarget,
      profileCall2Note: prof.buildProfileCall2Note(profile), visualDirective: {} }),
    call3: prompts.buildSectionCall3Prompt({ thesis: th, chapter, section, chunk1, chunk2, usedConcepts, wordsTarget,
      profileCall3Note: prof.buildProfileCall3Note(profile), visualDirective: {} }),
  }
})

const titles = ['Chapter 1: Introduction', 'Literature Review', 'Theoretical Framework', 'Research Methodology', 'Research Design',
  'Data Collection', 'Findings', 'Results and Discussion', 'Data Analysis', 'Discussion', 'Conclusion', 'Implications',
  'Contributions', 'Recommendations', 'Limitations', 'Appendix A', 'Appendices', 'References', 'Acknowledgements',
  'Background of the Study', 'Conceptual Framework', 'Theoretical Background', 'Research Approach', 'Summary', 'Abstract']
const classify = titles.map(t => [t, prof.detectProfile(t, 99).id, prof.classifyTitle ? prof.classifyTitle(t) : undefined])

writeFileSync(path.join(here, 'cases.json'), JSON.stringify({ cases, titles: classify.map(([t, d]) => [t, d]) }, null, 1))
console.log('wrote', cases.length, 'cases from', src)
