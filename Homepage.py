import streamlit as st

st.set_page_config(
    page_title="Main Page"
)

st.title("Dynamic Topic Modeling from PDF")
st.sidebar.success("Select a page above")

st.markdown(
    """
    <p style="text-align: center; font-size: 1.2em;">
        Upload your PDFs, analyze the content, and discover hidden topics.
    </p>
    """, 
    unsafe_allow_html=True
)

# Dynamic Topic Modeling description
st.header("What is Dynamic Topic Modeling?")
st.write("""
Dynamic Topic Modeling is a powerful approach for uncovering latent themes in large collections of text. 
By extracting and analyzing topics from uploaded PDFs, this tool provides a deep understanding of the underlying content. 
Leveraging advanced algorithms like **LDA (Latent Dirichlet Allocation)** and **NMF (Non-Negative Matrix Factorization)**, 
it delivers insights with precision and flexibility.
""")

st.subheader("Techniques Used")
st.markdown(
    """
    ### **1. Latent Dirichlet Allocation (LDA)**
    LDA is a probabilistic model that identifies topics as distributions over words and documents as mixtures of topics.
    - **Advantages**:
        - Probabilistic Nature: Captures the uncertainty in topic-word and document-topic distributions.
        - Interpretable Results: Generates human-understandable topics with strong semantic coherence.
        - Scalability: Handles large datasets efficiently.
        - Customization: Parameters like the number of topics can be tailored for specific datasets.

    ### **2. Non-Negative Matrix Factorization (NMF)**
    NMF decomposes a document-term matrix into two lower-dimensional matrices representing topics and their relationships with words and documents.
    - **Advantages**:
        - Deterministic Outputs: Produces consistent results for the same input data.
        - Simplicity: Does not require probabilistic assumptions.
        - Topic Specificity: Generates sharper and more distinct topics compared to LDA in some scenarios.
        - Efficient Computation: Often faster than LDA for small-to-medium datasets.
    """
)

st.subheader("Why Choose Dynamic Topic Modeling?")
st.write("""
- **Uncover Hidden Patterns**: Automatically detect recurring themes and trends in text data.
- **Flexible Text Analysis**: Works with a variety of documents, from academic papers to business reports.
- **Customizable Workflow**: Tailor the number of topics and level of detail to your needs.
- **Rich Visualization**: Provides clear, interactive topic modeling results for enhanced comprehension.
- **Effortless Automation**: Streamline the process of text analysis without manual intervention.
""")

st.markdown(
    """
    With Dynamic Topic Modeling, you gain actionable insights into your PDF documents, empowering you to make data-driven decisions 
    and understand your content at a deeper level. Choose between **LDA** for probabilistic exploration or **NMF** for deterministic 
    and distinct topics—whichever best suits your analysis needs!
    """,
    unsafe_allow_html=True
)
