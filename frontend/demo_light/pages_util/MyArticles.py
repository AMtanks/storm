import os
import shutil
import streamlit as st
from streamlit_card import card

import demo_util
from demo_util import DemoFileIOHelper, DemoUIHelper


# set page config and display title
def my_articles_page():
    demo_util.clear_other_page_session_state(page_index=2)

    with st.sidebar:
        # Initialize the session state for version toggling
        if 'show_polished_version' not in st.session_state:
            st.session_state.show_polished_version = True

        # Check if both versions of the article exist
        polished_exists = False
        raw_exists = False
        if "page2_selected_my_article" in st.session_state:
            article_name = st.session_state["page2_selected_my_article"]
            paths = st.session_state.get("page2_user_articles_file_path_dict", {}).get(article_name, {})
            polished_exists = "storm_gen_article_polished.txt" in paths
            raw_exists = "storm_gen_article.txt" in paths

        # Create columns for the buttons
        button_cols = st.columns([1, 1])
        
        with button_cols[1]: # "Select another article" button on the right
            if st.button(
                "Select another article",
                disabled="page2_selected_my_article" not in st.session_state,
                use_container_width=True
            ):
                if "page2_selected_my_article" in st.session_state:
                    del st.session_state["page2_selected_my_article"]
                st.session_state.show_polished_version = True # Reset to default when changing articles
                st.rerun()

        with button_cols[0]: # "Toggle version" button on the left
            if polished_exists and raw_exists:
                button_text = "View Raw Version" if st.session_state.show_polished_version else "View Polished Version"
                if st.button(button_text, use_container_width=True):
                    st.session_state.show_polished_version = not st.session_state.show_polished_version
                    st.rerun()

    # sync my articles
    if "page2_user_articles_file_path_dict" not in st.session_state:
        local_dir = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR")
        os.makedirs(local_dir, exist_ok=True)
        st.session_state["page2_user_articles_file_path_dict"] = (
            DemoFileIOHelper.read_structure_to_dict(local_dir)
        )

    # if no feature demo selected, display all featured articles as info cards
    def article_card_setup(column_to_add, card_title, article_name):
        with column_to_add:
            # Inject CSS for the custom button style
            st.markdown("""
                <style>
                    .stButton>button {
                        color: #808080;
                        background-color: transparent;
                        border: none;
                    }
                    .stButton>button:hover {
                        color: #f63366;
                        background-color: transparent;
                        border: none;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([0.9, 0.1])

            with col1:
                hasClicked = card(
                    title=" / ".join(card_title),
                    text=article_name.replace("_", " "),
                    image=DemoFileIOHelper.read_image_as_base64(
                        os.path.join(demo_util.get_demo_dir(), "assets", "void.jpg")
                    ),
                    styles=DemoUIHelper.get_article_card_UI_style(boarder_color="#9AD8E1"),
                )
                if hasClicked:
                    st.session_state["page2_selected_my_article"] = article_name
                    st.rerun()

            with col2:
                st.write("") # Vertical alignment
                if st.button("×", key=f"delete_{article_name}", help="Delete article"):
                    folder_to_delete = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR", article_name)
                    if os.path.isdir(folder_to_delete):
                        try:
                            shutil.rmtree(folder_to_delete)
                            # Refresh the list
                            st.session_state["page2_user_articles_file_path_dict"] = DemoFileIOHelper.read_structure_to_dict(
                                os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR")
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting: {e}")

    if "page2_selected_my_article" not in st.session_state:
        # get article names
        local_dir = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR")
        article_folders = list(st.session_state["page2_user_articles_file_path_dict"].keys())
        
        # Sort folders by modification time (newest first)
        try:
            article_names = sorted(
                article_folders,
                key=lambda name: os.path.getmtime(os.path.join(local_dir, name)),
                reverse=True
            )
        except FileNotFoundError:
            # Handle cases where a folder was deleted but session_state is not updated yet
            st.rerun()

        # display article cards
        my_article_columns = st.columns(3)
        if len(st.session_state["page2_user_articles_file_path_dict"]) > 0:
            # configure pagination
            pagination = st.container()
            bottom_menu = st.columns((1, 4, 1, 1, 1))[1:-1]
            with bottom_menu[2]:
                batch_size = st.selectbox("Page Size", options=[24, 48, 72])
            with bottom_menu[1]:
                total_pages = (
                    int(len(article_names) / batch_size)
                    if int(len(article_names) / batch_size) > 0
                    else 1
                )
                current_page = st.number_input(
                    "Page", min_value=1, max_value=total_pages, step=1
                )
            with bottom_menu[0]:
                st.markdown(f"Page **{current_page}** of **{total_pages}** ")
            # show article cards
            with pagination:
                my_article_count = 0
                start_index = (current_page - 1) * batch_size
                end_index = min(current_page * batch_size, len(article_names))
                for article_name in article_names[start_index:end_index]:
                    column_to_add = my_article_columns[my_article_count % 3]
                    my_article_count += 1
                    article_card_setup(
                        column_to_add=column_to_add,
                        card_title=["My Article"],
                        article_name=article_name,
                    )
        else:
            with my_article_columns[0]:
                hasClicked = card(
                    title="Get started",
                    text="Start your first research!",
                    image=DemoFileIOHelper.read_image_as_base64(
                        os.path.join(demo_util.get_demo_dir(), "assets", "void.jpg")
                    ),
                    styles=DemoUIHelper.get_article_card_UI_style(),
                )
                if hasClicked:
                    st.session_state.selected_page = 1
                    st.session_state["manual_selection_override"] = True
                    st.session_state["rerun_requested"] = True
                    st.rerun()
    else:
        selected_article_name = st.session_state["page2_selected_my_article"]
        selected_article_file_path_dict = st.session_state[
            "page2_user_articles_file_path_dict"
        ][selected_article_name]

        demo_util.display_article_page(
            selected_article_name=selected_article_name,
            selected_article_file_path_dict=selected_article_file_path_dict,
            show_title=True,
            show_main_article=True,
        )
