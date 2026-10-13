import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape
from io import BytesIO

st.set_page_config(page_title='Smart Scheme Guide', page_icon='🏛️', layout='wide')

SCHEMES = [
    dict(id='pmkisan', name='PM-KISAN Samman Nidhi', ta='பிரதமரின் கிசான் சம்மான் நிதி', tg='PM-KISAN Thittam', category='Farmers', benefit='₹6,000 per year in three instalments, subject to current rules.', eligibility='Eligible landholding farmer families, subject to exclusions.', documents='Aadhaar, land records, bank details', link='https://pmkisan.gov.in/', keywords='farmer farming vivasayi விவசாயி agriculture land nilam', tags=['farmer']),
    dict(id='pmjay', name='Ayushman Bharat PM-JAY', ta='ஆயுஷ்மான் பாரத் மருத்துவத் திட்டம்', tg='Ayushman Bharat Maruthuva Thittam', category='Health', benefit='Health cover up to ₹5 lakh per eligible family per year.', eligibility='Families covered by the programme criteria; check the official portal.', documents='Identity and eligibility documents as required by the portal', link='https://pmjay.gov.in/', keywords='health insurance hospital medical maruthuvam மருத்துவம் kaappeedu', tags=['health']),
    dict(id='penn', name='Pudhumai Penn Scheme', ta='புதுமைப் பெண் திட்டம்', tg='Pudhumai Penn Thittam', category='Students', benefit='Monthly education assistance of ₹1,000 under applicable rules.', eligibility='Eligible female higher-education students with qualifying school background; verify current rules.', documents='Aadhaar, school/college details, bank details', link='https://www.pudhumaipenn.tn.gov.in/', keywords='girl woman student college education scholarship maanavi மாணவி padikuren pg ug', tags=['student','woman']),
    dict(id='magalir', name='Kalaignar Magalir Urimai Thogai', ta='கலைஞர் மகளிர் உரிமைத் தொகை', tg='Kalaignar Magalir Urimai Thogai', category='Women', benefit='Monthly assistance for eligible women heads of households, subject to current rules.', eligibility='Women heads of qualifying Tamil Nadu households; income and asset rules apply.', documents='Ration card, Aadhaar, bank details and other required proofs', link='https://kmut.tn.gov.in/', keywords='women woman family head magalir pengal பெண்கள் kudumba thalaivi', tags=['woman']),
    dict(id='cmchis', name="Chief Minister's Comprehensive Health Insurance Scheme", ta='முதலமைச்சரின் விரிவான மருத்துவக் காப்பீட்டுத் திட்டம்', tg='Mudhalamaichar Maruthuva Kaappeedu', category='Health', benefit='Cashless treatment for covered procedures within current scheme limits.', eligibility='Qualifying Tamil Nadu families; check the latest income and other conditions.', documents='Family card, income proof and identity documents as required', link='https://www.cmchistn.com/', keywords='health medical hospital tamil nadu insurance sigichai சிகிச்சை', tags=['health']),
    dict(id='tamilpudhalvan', name='Tamil Pudhalvan Scheme', ta='தமிழ்ப் புதல்வன் திட்டம்', tg='Tamil Pudhalvan Thittam', category='Students', benefit='Monthly education assistance under the programme rules.', eligibility='Eligible male higher-education students with qualifying school background; check current criteria.', documents='Education records, Aadhaar and bank details', link='https://www.tn.gov.in/', keywords='boy male student college education scholarship maanavan மாணவன் pg ug padikuren', tags=['student','man']),
    dict(id='pmfby', name='Pradhan Mantri Fasal Bima Yojana', ta='பிரதம மந்திரி பயிர் காப்பீட்டுத் திட்டம்', tg='Payir Kaappeedu Thittam', category='Farmers', benefit='Crop insurance protection subject to crop, season and enrolment rules.', eligibility='Farmers growing notified crops in notified areas and seasons.', documents='Land/cultivation proof, identity, bank and crop details', link='https://pmfby.gov.in/', keywords='farmer crop insurance farming vivasayi payir பயிர்', tags=['farmer']),
    dict(id='pmkvy', name='Pradhan Mantri Kaushal Vikas Yojana', ta='பிரதம மந்திரி திறன் மேம்பாட்டுத் திட்டம்', tg='Thiran Membattu Thittam', category='Employment', benefit='Eligible skill-training and certification opportunities.', eligibility='Depends on course, training provider and current programme guidelines.', documents='Identity, education and enrolment documents as applicable', link='https://www.skillindiadigital.gov.in/', keywords='job employment unemployed skill training velai வேலை', tags=['job','student']),
    dict(id='mudra', name='Pradhan Mantri MUDRA Yojana', ta='பிரதம மந்திரி முத்ரா திட்டம்', tg='MUDRA Kadan Thittam', category='Business', benefit='Business loans through participating lenders, subject to approval.', eligibility='Qualifying micro-enterprises and eligible applicants; lender checks apply.', documents='Identity, address, business and financial documents as required', link='https://www.mudra.org.in/', keywords='business self employment entrepreneur startup loan kadan தொழில்', tags=['business']),
    dict(id='pmay', name='Pradhan Mantri Awas Yojana', ta='பிரதம மந்திரி வீட்டு வசதித் திட்டம்', tg='Veedu Vasathi Thittam', category='Housing', benefit='Housing assistance under the applicable urban or rural programme.', eligibility='Depends on location, household and current scheme conditions.', documents='Identity, household, income and property-related proofs as applicable', link='https://pmay-urban.gov.in/', keywords='house home housing veedu வீடு', tags=['housing']),
]

LANGS = {'English':'en', 'தமிழ்':'ta', 'Tanglish':'tg'}
WORDS = {
 'en': {'home':'Dashboard','search':'Find Schemes','recommend':'Recommendations','check':'Eligibility Checker','saved':'Saved Schemes','faq':'FAQ Assistant','search_label':'Describe your need','category':'Category','all':'All','benefit':'Benefits','eligible':'Eligibility','docs':'Documents','official':'Official website','save':'Save','unsave':'Remove bookmark','none':'No matching schemes found. Try another query or category.'},
 'ta': {'home':'முகப்பு','search':'திட்டங்களைத் தேடுங்கள்','recommend':'பரிந்துரைகள்','check':'தகுதி சரிபார்ப்பு','saved':'சேமித்த திட்டங்கள்','faq':'கேள்வி பதில்கள்','search_label':'உங்கள் தேவையை எழுதுங்கள்','category':'வகை','all':'அனைத்தும்','benefit':'பலன்கள்','eligible':'தகுதி','docs':'ஆவணங்கள்','official':'அதிகாரப்பூர்வ இணையதளம்','save':'சேமி','unsave':'நீக்கு','none':'பொருத்தமான திட்டம் கிடைக்கவில்லை. வேறு வார்த்தைகளை முயற்சிக்கவும்.'},
 'tg': {'home':'Mugappu','search':'Thittangalai Thedungal','recommend':'Parindhuraigal','check':'Thagudhi Check','saved':'Save Panna Thittangal','faq':'Kelvi Bathil','search_label':'Ungal thevaiyai type pannungal','category':'Vagai','all':'Anaithum','benefit':'Palangal','eligible':'Thagudhi','docs':'Aavanangal','official':'Official Website','save':'Save','unsave':'Remove','none':'Poruthamaana thittam illai. Vera words try pannunga.'}}

if 'saved' not in st.session_state: st.session_state.saved = []
if 'profile' not in st.session_state: st.session_state.profile = {}

@st.cache_resource
def index():
    corpus = [' '.join([s['name'],s['ta'],s['tg'],s['category'],s['benefit'],s['eligibility'],s['keywords']]).lower() for s in SCHEMES]
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4))
    return vec, vec.fit_transform(corpus)

def matches(query, category='All'):
    ids = [i for i,s in enumerate(SCHEMES) if category == 'All' or s['category'] == category]
    if not query.strip(): return [SCHEMES[i] for i in ids]
    vec, matrix = index()
    scores = cosine_similarity(vec.transform([query.strip().lower()]),matrix)[0]
    return [SCHEMES[i] for i in sorted(ids,key=lambda i:scores[i],reverse=True) if scores[i] > .08]

def pdf_bytes(s):
    buf=BytesIO(); doc=SimpleDocTemplate(buf); styles=getSampleStyleSheet()
    story=[Paragraph(escape(s['name']),styles['Title']),Spacer(1,16)]
    for heading, value in [('Category',s['category']),('Benefits',s['benefit']),('Eligibility (indicative)',s['eligibility']),('Documents',s['documents']),('Official website',s['link'])]:
        story.append(Paragraph(f'<b>{escape(heading)}:</b> {escape(value)}',styles['BodyText'])); story.append(Spacer(1,10))
    story.append(Paragraph('Information is illustrative. Verify all conditions on the official website before applying.',styles['Italic']))
    doc.build(story); return buf.getvalue()

def card(s,lang,t):
    with st.container(border=True):
        st.subheader(s['ta'] if lang=='ta' else s['tg'] if lang=='tg' else s['name'])
        st.caption(s['category'])
        st.write(f"**{t['benefit']}:** {s['benefit']}")
        with st.expander(t['eligible']+' / '+t['docs']):
            st.write(s['eligibility']); st.write('**Documents:** '+s['documents'])
        a,b,c=st.columns(3)
        with a: st.link_button('🔗 '+t['official'],s['link'],use_container_width=True)
        with b:
            is_saved=s['id'] in st.session_state.saved
            if st.button('❤️ '+(t['unsave'] if is_saved else t['save']),key='save_'+s['id']):
                if is_saved: st.session_state.saved.remove(s['id'])
                else: st.session_state.saved.append(s['id'])
                st.rerun()
        with c: st.download_button('📄 PDF',pdf_bytes(s),file_name=s['id']+'.pdf',mime='application/pdf',key='pdf_'+s['id'])

st.sidebar.title('🏛️ Smart Scheme Guide')
lang=LANGS[st.sidebar.radio('🌐 Language',list(LANGS),horizontal=False)]
t=WORDS[lang]
sections=[('home','📊'),('search','🔎'),('recommend','🎯'),('check','✅'),('saved','❤️'),('faq','💬')]
choice=st.sidebar.radio('Menu',[key for key,_ in sections],format_func=lambda k:dict(sections)[k]+' '+t[k])
st.sidebar.caption('Sample information only. Check current official rules before applying. Bookmarks are kept for this browser session.')
st.title('🏛️ Smart Government Scheme Guide')
st.caption('Tamil • English • Tanglish | Search, explore and shortlist schemes')

if choice=='home':
    a,b,c=st.columns(3)
    a.metric('Schemes',len(SCHEMES)); b.metric('Categories',len(set(s['category'] for s in SCHEMES))); c.metric('Saved',len(st.session_state.saved))
    st.subheader('Category overview')
    counts=pd.Series([s['category'] for s in SCHEMES]).value_counts()
    st.bar_chart(counts)
    st.info('Choose Find Schemes to search, or Recommendations to enter your profile.')
elif choice=='search':
    q=st.text_input(t['search_label'],placeholder='e.g. naa PG padikuren scholarship venum')
    cats=['All']+sorted(set(s['category'] for s in SCHEMES))
    cat=st.selectbox(t['category'],cats,format_func=lambda x:t['all'] if x=='All' else x)
    found=matches(q,cat)
    st.write(f'**{len(found)} results**')
    if not found: st.warning(t['none'])
    for s in found: card(s,lang,t)
elif choice=='recommend':
    st.subheader('Profile-based recommendations')
    st.caption('Broad topic matching only — this is not a legal eligibility determination.')
    with st.form('profile_form'):
        age=st.number_input('Age',min_value=1,max_value=110,value=22)
        gender=st.selectbox('Gender (optional for broad matching)',['Prefer not to say','Female','Male','Other'])
        role=st.selectbox('I am a',['Student','Farmer','Job seeker','Business owner','Other'])
        need=st.multiselect('Interested in',['Education','Health','Employment','Business','Housing','Farming'])
        submitted=st.form_submit_button('Find recommendations')
    if submitted: st.session_state.profile=dict(age=age,gender=gender,role=role,need=need)
    if st.session_state.profile:
        p=st.session_state.profile
        tagmap={'Student':'student','Farmer':'farmer','Job seeker':'job','Business owner':'business'}
        needmap={'Education':'student','Health':'health','Employment':'job','Business':'business','Housing':'housing','Farming':'farmer'}
        tags=set([tagmap.get(p['role'])]+[needmap[n] for n in p['need']]); tags.discard(None)
        if p['gender']=='Female': tags.add('woman')
        ranked=sorted(SCHEMES,key=lambda s:len(tags.intersection(s['tags'])),reverse=True)
        ranked=[s for s in ranked if tags.intersection(s['tags'])]
        if not ranked: st.info('Select an interest to get recommendations.')
        for s in ranked: card(s,lang,t)
elif choice=='check':
    st.subheader('Eligibility guidance')
    selected=st.selectbox('Choose a scheme',[s['id'] for s in SCHEMES],format_func=lambda x:next(s['name'] for s in SCHEMES if s['id']==x))
    s=next(s for s in SCHEMES if s['id']==selected)
    st.write('**Published-rule summary (illustrative):**',s['eligibility'])
    st.write('**Documents to prepare:**',s['documents'])
    with st.form('eligibility_form'):
        residence=st.selectbox('Residence',['Tamil Nadu','Other Indian state','Outside India'])
        status=st.selectbox('Does the scheme description appear to match your situation?',['Not sure','Yes','No'])
        submitted=st.form_submit_button('Check guidance')
    if submitted:
        if status=='No': st.warning('The stated conditions may not match your situation. Check the official portal for exceptions or other schemes.')
        else: st.info('Possible match / further verification required. Residence alone does not establish eligibility; official criteria and documents decide.')
        st.link_button('Verify on official portal',s['link'])
elif choice=='saved':
    chosen=[s for s in SCHEMES if s['id'] in st.session_state.saved]
    if not chosen: st.info('No bookmarks yet. Save a scheme from Find Schemes.')
    for s in chosen: card(s,lang,t)
elif choice=='faq':
    st.subheader('Frequently asked questions')
    faqs=[('How can I apply?','Open the official website from the relevant scheme card and follow its current instructions.'),('Are recommendations guaranteed?','No. They are topic matches, not confirmation of eligibility or approval.'),('Which documents do I need?','Open a scheme card to see an indicative list, then confirm it on the official portal.'),('Can I search in Tanglish?','Yes. Try words such as maanavi, vivasayi, maruthuvam, padikuren or scholarship.'),('Will my saved schemes remain forever?','No. Bookmarks are stored in the current Streamlit session only.')]
    for q,a in faqs:
        with st.expander(q): st.write(a)

st.divider()
st.caption('⚠️ Demonstration dataset. Scheme rules, benefits, links and availability may change. Always verify details on official government portals before applying.')
