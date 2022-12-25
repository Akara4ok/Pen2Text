import React from 'react';
import classes from './TextEditor.scss';
import Button from '@Components/Button/Button';
import DropdownList from '@Components/DropdownList/DropdownList';

class TextEditor extends React.Component {
    constructor(props) {
        super(props);
    }
    render() {
        return (
            <div className={classes.wrapper}>
                Text
                <DropdownList
                    className={classes.dropdownStyle}
                    items={['*.txt', '*.docx', '*.pdf']}
                />
                <Button className={classes.buttonStyle}>Download</Button>
            </div>
        );
    }
}

export default TextEditor;
