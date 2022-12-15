import React from 'react';
import classes from './PenEditor.scss';
import Button from '@Components/Button/Button';
import DropdownList from '@Components/DropdownList/DropdownList';

class PenEditor extends React.Component {
    constructor(props) {
        super(props);
    }
    render() {
        return (
            <div className={classes.wrapper}>
                <div>Draw</div>
                <DropdownList />
                <Button className={classes.buttonStyle}>To Uploader</Button>
            </div>
        );
    }
}

export default PenEditor;
